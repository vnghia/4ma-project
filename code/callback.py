class CallbackLogger:
    def __init__(self, object):
        self.number_iteration = 1
        self.object = object

    def __call__(self, xk):
        self.number_iteration += 1
