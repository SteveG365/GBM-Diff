from abc import ABC, abstractmethod

class DiffusionBackbone(ABC):

    @abstractmethod
    def predict_noise(
        self,
        x_t,
        timestep,
        control_state = None):

        raise NotImplementedError
