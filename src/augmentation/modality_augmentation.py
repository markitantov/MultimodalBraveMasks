import torch

class ModalityDropAugmentation(torch.nn.Module):
    """Randomly dropes one of the modality zeroing out all elements
    Generates value (uniform distribution) on specified limits
    
    Logic:
    if generated_value in limits[0]: drop audio modality
    if generated_value in limits[1]: do nothing
    if generated_value in limits[2]: drop video modality
    
    Args:
        limits (list[tuple[int, int]], optional): Limits of generated value. Defaults to [(0, 20), (20, 80), (80, 100)].
    """
    def __init__(self, limits: list[tuple[int, int]] = None) -> None:
        super(ModalityDropAugmentation, self).__init__()
        self.limits = limits if limits else [(0, 20), (20, 80), (80, 100)]
        self.min_l = self.limits[0][0]
        self.max_l = self.limits[2][1]
    
    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Generates value (uniform distribution) on specified limits
        and drop (zeroing out) modalities

        Args:
            x (list[torch.Tensor]): Input (audio, video) tensor

        Returns:
            list[torch.Tensor]: Modified (audio, video) tensor
        """
        a, v = x
        # generate uniformly distributed value on [min_l, max_l].
        choise = torch.FloatTensor(1).uniform_(self.min_l, self.max_l)
        if self.limits[0][0] <= choise < self.limits[0][1]:
            a = torch.zeros(a.shape)
        elif self.limits[1][0] <= choise < self.limits[1][1]:
            return a, v
        elif self.limits[2][0] <= choise <= self.limits[2][1]:
            v = torch.zeros(v.shape)
        
        return a, v