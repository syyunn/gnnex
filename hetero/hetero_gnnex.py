import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.data import HeteroData
from torch_geometric.explain import ExplainerConfig, Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_masks
)
# import Optional
from typing import Optional, Union, Dict, Tuple
from model import BuySellLinkPrediction

from torch_geometric.nn import MessagePassing
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel


def set_hetero_masks(
    model: torch.nn.Module,
    mask_dict: Dict[Tuple[str, str, str], Union[Tensor, Parameter]],
    edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    apply_sigmoid: bool = True,
):
    r"""Apply masks to every heterogeneous graph layer in the :obj:`model`
    according to edge types."""

    def set_edge_type_masks(layer, mask, edge_index, apply_sigmoid):
        if isinstance(layer, MessagePassing):
            if (not isinstance(mask, Parameter) and '_edge_mask' in layer._parameters):
                mask = Parameter(mask)
            layer.explain = True # this sends the layer to explain part
            layer._edge_mask = mask
            layer._loop_mask = edge_index[0] != edge_index[1]
            layer._apply_sigmoid = apply_sigmoid

    for module in model.modules():
        if isinstance(module, torch.nn.ModuleDict):
            for edge_type, mask in mask_dict.items():
                str_edge_type = '__'.join(edge_type)
                if str_edge_type in module:
                    edge_index = edge_index_dict[edge_type]
                    layer = module[str_edge_type]
                    set_edge_type_masks(layer, mask, edge_index, apply_sigmoid)
        # elif isinstance(module, MessagePassing):
        #     for edge_type, mask in mask_dict.items():
        #         edge_index = edge_index_dict[edge_type]
        #         set_edge_type_masks(module, mask, edge_index, apply_sigmoid)


class HeteroGNNExplainer(ExplainerAlgorithm):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.1, device='cpu', data=None, edge_label_index=None, edge_label_attr=None, l1_lambda=None, **kwargs):
        super().__init__()
        self.edge_label_index = edge_label_index
        self.edge_label_attr = edge_label_attr
        self.data=data
        self.device = device
        self.model=model
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)
        self.l1_lambda = l1_lambda

        self.node_masks = {}
        self.hard_node_masks = {}
        self.edge_masks = {}
        self.hard_edge_masks = {}

    def get_initial_prediction(
        self,
        data: HeteroData,
        edge_index_to_explain: torch.Tensor,
        target_node_type: str,
        target_edge_type: str,
        model_path: str = "buysell_link_prediction_best_model.pt",
    ) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assign consecutive indices to each node type
        for node_type in data.node_types:
            data[node_type].node_id = torch.arange(data[node_type].num_nodes)

        # Convert edge_index tensors to integer type (torch.long)
        for edge_type, edge_index in data.edge_index_dict.items():
            data.edge_index_dict[edge_type] = edge_index.to(torch.long)

        num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}
        num_layers = 2
        edge_types = list(data.edge_index_dict.keys())

        model = BuySellLinkPrediction(
            num_nodes_dict,
            embedding_dim=64,
            num_edge_features=2,
            out_channels=64,
            edge_types=edge_types,
            num_layers=num_layers,
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # date scaling
        from datetime import date
        start_date = date(2016, 1, 1)
        today = date.today()
        total_days = (today - start_date).days

        scaled_edge_attr_dict = {key: value / total_days for key, value in data.edge_attr_dict.items()}
        x_dict = {node_type: data[node_type].node_id for node_type in num_nodes_dict.keys()}

        # Create edge_label_attr tensor
        key = (target_node_type, target_edge_type)
        edge_attr = data.edge_attr_dict[key][edge_index_to_explain]
        edge_attr = torch.tensor(edge_attr / total_days, dtype=torch.float, device=device)

        with torch.no_grad():
            preds = model(x_dict, data.edge_index_dict, scaled_edge_attr_dict, edge_index_to_explain, edge_label_attr=edge_attr)

        return preds.item()
    
    def _train(
            self,
            model: torch.nn.Module,
            x_dict: Dict[str, Tensor],
            edge_index_dict: Dict[str, Tensor],
            *,
            target: Tensor,
            index: Optional[Union[int, Tensor]] = None,
            **kwargs
        ):
        self._initialize_masks(x_dict, edge_index_dict)

        parameters = []
        if self.node_mask_dict is not None:
            parameters.extend(self.node_mask_dict.values())
        if self.edge_mask_dict is not None:
            # this is the part we assign edge mask to all layers
            set_hetero_masks(model, self.edge_mask_dict, edge_index_dict, apply_sigmoid=True)
            parameters.extend(self.edge_mask_dict.values())

        # Define the learning rate scheduler function
        def custom_lr_schedule(epoch):
            if epoch < 3:
                return 10
            elif epoch < 10:
                return 1
            elif epoch < 20:
                return 0.1
            else:
                return 0.01

        # Set up the optimizer
        optimizer = torch.optim.Adam(parameters, lr=self.lr) # this means we're udpating this edge/node_mask dict only.

        # Set up the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_lr_schedule)


        for i in range(self.epochs):            
            # print("Epoch: ", i)
            optimizer.zero_grad()

            h_dict = {
                key: x.weight if self.node_mask_dict is None or key not in self.node_mask_dict
                else x.weight * self.node_mask_dict[key].sigmoid()
                for key, x in x_dict.items()
            }
            # preds = model(x_dict, data.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=edge_attr)

            # date scaling
            from datetime import date
            start_date = date(2016, 1, 1)
            today = date.today()
            total_days = (today - start_date).days
            scaled_edge_attr_dict = {key: value / total_days for key, value in self.data.edge_attr_dict.items()}

            _, y_hat = model(h_dict, edge_index_dict, scaled_edge_attr_dict, self.edge_label_index, self.edge_label_attr)
            y = target

            # print("y_hat", y_hat)
            # print("y", y)

            if index is not None:
                y_hat, y = y_hat[index], y[index]
            
            loss = self._loss(y_hat, y, self.l1_lambda)
            print("loss", loss)
            print("y_hat", y_hat)
            print("y", y)

            loss.backward()
            optimizer.step()

            scheduler.step()

            if i == 0:
                if self.node_mask_dict is not None:
                    self.hard_node_mask_dict = {
                        key: mask.grad != 0.0 for key, mask in self.node_mask_dict.items()
                    }
                if self.edge_mask_dict is not None:
                    self.hard_edge_mask_dict = {
                        key: mask.grad != 0.0 for key, mask in self.edge_mask_dict.items()
                    }

    def _initialize_masks(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[str, Tensor]):
        node_mask_type = 'object'
        edge_mask_type = 'object'

        device = self.device

        std = 0.1
        if node_mask_type is None:
            self.node_mask_dict = None
        else:
            self.node_mask_dict = {}
            for key, x in x_dict.items():
                N = x.num_embeddings
                F = None # not implemnted

                if node_mask_type == 'object':
                    self.node_mask_dict[key] = Parameter(torch.randn(N, 1, device=device) * std) # we are using random initalization.
                elif node_mask_type == MaskType.attributes:
                    self.node_mask_dict[key] = Parameter(torch.randn(N, F, device=device) * std)
                elif node_mask_type == MaskType.common_attributes:
                    self.node_mask_dict[key] = Parameter(torch.randn(1, F, device=device) * std)
                else:
                    assert False

        if edge_mask_type is None:
            self.edge_mask_dict = None
        else:
            self.edge_mask_dict = {}
            for key, edge_index in edge_index_dict.items():
                E = edge_index.size(1)
                if edge_mask_type == 'object':
                    from math import sqrt
                    std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * E))
                    self.edge_mask_dict[key] = Parameter(torch.randn(E, device=device) * std) # this corresponds to the edge size! 
                else:
                    assert False

    def explain_edge(
        self,
        edge_index_to_explain: Tuple[int, int],
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
        **kwargs):

        source_node, target_node = edge_index_to_explain
        explanation = self._explainer(
            self.model,
            x_dict,
            edge_index_dict,
            target=self.get_initial_prediction(x_dict, edge_index_dict, **kwargs),
            index=torch.tensor([source_node, target_node], device=list(x_dict.values())[0].device),
            **kwargs,
        )
        return self._convert_output(explanation, edge_index_dict, index=edge_index_to_explain, x_dict=x_dict)

    def _convert_output(self, explanation, edge_index_dict, index=None, x_dict=None):
        node_mask_dict = explanation.get('node_mask_dict')
        edge_mask_dict = explanation.get('edge_mask_dict')

        if node_mask_dict is not None:
            node_mask_type = self._explainer.explainer_config.node_mask_type
            if node_mask_type in {MaskType.object, MaskType.common_attributes}:
                for key, mask in node_mask_dict.items():
                    node_mask_dict[key] = mask.view(-1)

        if edge_mask_dict is None:
            if index is not None:
                _, edge_mask_dict = self._explainer._get_hard_masks(
                    self.model, index, edge_index_dict, num_nodes_dict={k: v.size(0) for k, v in x_dict.items()})
                for key, mask in edge_mask_dict.items():
                    edge_mask_dict[key] = mask.to(x_dict[key].dtype)
            else:
                edge_mask_dict = {k: torch.ones(edge_index.shape[1], device=edge_index.device) for k, edge_index in edge_index_dict.items()}

        return node_mask_dict, edge_mask_dict


    def forward(
        self,
        model: torch.nn.Module,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        self._train(model, x_dict, edge_index_dict, target=target, index=index, **kwargs)

        node_masks = {}
        for key in self.node_mask_dict:
            node_masks[key] = self._post_process_mask(
                self.node_mask_dict[key],
                self.hard_node_mask_dict[key],
                apply_sigmoid=True,
            )
        edge_masks = {}
        for key in self.edge_mask_dict:
            edge_masks[key] = self._post_process_mask(
                self.edge_mask_dict[key],
                self.hard_edge_mask_dict[key],
                apply_sigmoid=True,
            )

        self._clean_model(model)

        return Explanation(node_masks=node_masks, edge_masks=edge_masks)


    def _clean_model(self, model: torch.nn.Module):
        def clear_hetero_masks(model: torch.nn.Module):
            r"""Clear all masks from the model."""
            for module in model.modules():
                if isinstance(module, torch.nn.ModuleDict):
                    for layer in module.values():
                        if isinstance(layer, MessagePassing):
                            layer.explain = False
                            layer._edge_mask = None
                            layer._loop_mask = None
                            layer._apply_sigmoid = True
                elif isinstance(module, MessagePassing):
                    module.explain = False
                    module._edge_mask = None
                    module._loop_mask = None
                    module._apply_sigmoid = True
            return module

        clear_hetero_masks(model)


    def _loss(self, y_hat: Tensor, y: Tensor, l1_lambda: float) -> Tensor:
        # loss = -torch.sum(y * torch.log(y_hat), dim=0).mean()
        loss = torch.mean(torch.sum((y - y_hat) ** 2, dim=0))

        for key in self.coeffs:
            if key in self.node_mask_dict:
                loss += self.coeffs[key] * self.node_mask_dict[key].sigmoid().sum()
            if key in self.edge_mask_dict:
                loss += self.coeffs[key] * self.edge_mask_dict[key].sigmoid().sum()

        # Add L1 regularization - this is for sparse explanability
        l1_regularization = torch.tensor(0.0, device=self.device)
        for param in self.parameters():
            l1_regularization += torch.norm(param, 1)
        loss += l1_lambda * l1_regularization

        return loss    
