class StepLimitExceeded(RuntimeError):
    pass

class StepLimiter:
    def __init__(self, max_steps: int):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.max_steps = max_steps

    def ensure_capacity(self, completed_steps: int):
        if completed_steps >= self.max_steps:
            raise StepLimitExceeded(f"Agent exceeded maximum steps ({self.max_steps})")
