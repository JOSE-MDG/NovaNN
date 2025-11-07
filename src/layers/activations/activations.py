from src.module.layer import Layer


class Activation(Layer):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__.lower()
        self.affect_init = True

    def get_init_key(self):
        if not self.affect_init:
            return None
        return self.name
