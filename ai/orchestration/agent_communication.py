"""In-process request/response communication between registered agents."""

from __future__ import annotations

from ai.agents.agent_messages import AgentMessage, AgentMessageType
from ai.agents.agent_registry import AgentRegistry
from ai.agents.specialist_contracts import SpecialistRequest


class AgentCommunicationError(RuntimeError):
    """Raised when an internal agent message cannot be delivered."""


class AgentCommunicationBus:
    """Delivers typed messages between agents through the AgentRegistry.

    This is an internal, synchronous A2A pattern. It is not the external
    Google A2A protocol and does not transfer Swarm execution control.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._messages: list[AgentMessage] = []

    @property
    def messages(self) -> list[AgentMessage]:
        """Return a copy of the communication log."""
        return list(self._messages)

    def request(
        self,
        *,
        sender: str,
        recipient: str,
        task_id: str,
        user_request: str,
        context: dict | None = None,
        constraints: dict | None = None,
    ) -> AgentMessage:
        """Send a task to another agent and return its structured response."""
        sender_name = sender.strip().lower()
        recipient_name = recipient.strip().lower()

        if sender_name == recipient_name:
            raise AgentCommunicationError("An agent cannot message itself.")

        # Validate both ends before recording or delivering the message.
        self._registry.get(sender_name)
        self._registry.get(recipient_name)

        request_message = AgentMessage(
            sender=sender_name,
            recipient=recipient_name,
            message_type=AgentMessageType.REQUEST,
            task_id=task_id,
            payload={
                "user_request": user_request,
                "context": context or {},
                "constraints": constraints or {},
            },
        )
        self._messages.append(request_message)

        result = self._registry.invoke(
            recipient_name,
            SpecialistRequest(
                task_id=task_id,
                user_request=user_request,
                context=context or {},
                constraints=constraints or {},
            ),
        )

        response_message = AgentMessage(
            correlation_id=request_message.correlation_id,
            in_reply_to=request_message.message_id,
            sender=recipient_name,
            recipient=sender_name,
            message_type=AgentMessageType.RESPONSE,
            task_id=task_id,
            payload={"result": result.model_dump(mode="json")},
        )
        self._messages.append(response_message)
        return response_message
