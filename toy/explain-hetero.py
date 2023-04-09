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

class HeteroGNNExplainer(ExplainerAlgorithm):
    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_masks = {}
        self.hard_node_masks = {}
        self.edge_masks = {}
        self.hard_edge_masks = {}

    def forward(self, model: torch.nn.Module, x_dict: Tensor, edge_index_dict: Tensor, *, target: Tensor, index: Optional[Union[int, Tensor]] = None, **kwargs,) -> Explanation:
        self._train(model, x_dict, edge_index_dict, target=target, index=index, **kwargs)

        node_masks = {}
        for key in self.node_masks:
            node_masks[key] = self._post_process_mask(
                self.node_masks[key],
                self.hard_node_masks[key],
                apply_sigmoid=True,
            )
        edge_masks = {}
        for key in self.edge_masks:
            edge_masks[key] = self._post_process_mask(
                self.edge_masks[key],
                self.hard_edge_masks[key],
                apply_sigmoid=True,
            )

        self._clean_model(model)

        return Explanation(node_masks=node_masks, edge_masks=edge_masks)

    def _train(self, model: torch.nn.Module, x_dict: Tensor, edge_index_dict: Tensor, *, target: Tensor, index: Optional[Union[int, Tensor]] = None, **kwargs,):
        self._initialize_masks(x_dict, edge_index_dict)

        parameters = []
        for key in self.node_masks:
            parameters.append(self.node_masks[key])
        for key in self.edge_masks:
            parameters.append(self.edge_masks[key])

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            for key in x_dict:
                if key in self.node_masks:
                    x_dict[key] = x_dict[key] * self.node_masks[key].sigmoid()
            set_masks(model, self.edge_masks, edge_index_dict, apply_sigmoid=True)

            y_hat, y = model(x_dict, edge_index_dict, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            if i == 0:
                for key in self.node_masks:
                    self.hard_node_masks[key] = self.node_masks[key].grad != 0.0
                for key in self.edge_masks:
                    self.hard_edge_masks[key] = self.edge_masks[key].grad != 0.0

    def _initialize_masks(self, x_dict: Tensor, edge_index_dict: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        for key in x_dict:
            if key not in self.node_masks:
                if node_mask_type == MaskType.ZERO:
                    self.node_masks[key] = torch.zeros_like(x_dict[key], requires_grad=True)
                elif node_mask_type == MaskType.RANDOM:
                    self.node_masks[key] = Parameter(torch.randn_like(x_dict[key]), requires_grad=True)
                else:
                    raise ValueError(f"Invalid node_mask_type: {node_mask_type}")

        for key in edge_index_dict:
            if key not in self.edge_masks:
                num_edges = edge_index_dict[key].size(1)
                if edge_mask_type == MaskType.ZERO:
                    self.edge_masks[key] = torch.zeros(num_edges, device=x_dict[key].device, requires_grad=True)
                elif edge_mask_type == MaskType.RANDOM:
                    self.edge_masks[key] = Parameter(torch.randn(num_edges, device=x_dict[key].device), requires_grad=True)
                else:
                    raise ValueError(f"Invalid edge_mask_type: {edge_mask_type}")

    def _clean_model(self, model: torch.nn.Module):
        clear_masks(model)

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = -torch.sum(y * torch.log(y_hat), dim=1).mean()

        for key in self.coeffs:
            if key in self.node_masks:
                loss += self.coeffs[key] * self.node_masks[key].sigmoid().sum()
            if key in self.edge_masks:
                loss += self.coeffs[key] * self.edge_masks[key].sigmoid().sum()

        return loss