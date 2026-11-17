from ai.harness.execution_controller import AgentDecision

class FakeHotelModel:

    def __init__(self):
        self.step = 0

    def decide(self, context):

        self.step += 1

        if self.step == 1:
            return AgentDecision(
                kind="tool",
                tool_name="search_hotels",
                tool_input={
                    "city": "Varanasi"
                }
            )

        return AgentDecision(
            kind="final",
            content="I found two hotels within your budget."
        )