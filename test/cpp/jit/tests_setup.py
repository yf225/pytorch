import torch


def testEvalModeForLoadedModule_setup():
    class Model(torch.jit.ScriptModule):
        def __init__(self):
            super(Model, self).__init__()
            self.dropout = torch.nn.Dropout(0.1)

        def forward(self, x):
            x = self.dropout(x)
            return x

    model = Model()
    model = model.train()
    model.save('dropout_model.pt')


def setup():
    testEvalModeForLoadedModule_setup()


if __name__ == "__main__":
    setup()
