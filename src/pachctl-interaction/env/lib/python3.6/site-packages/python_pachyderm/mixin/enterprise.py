from python_pachyderm.service import Service


class EnterpriseMixin:
    def activate_enterprise(self, activation_code, expires=None):
        """
        Activates enterprise. Returns a `TokenInfo` object.

        Params:

        * `activation_code`: A string specifying a Pachyderm enterprise
        activation code. New users can obtain trial activation codes.
        * `expires`: An optional `Timestamp` object indicating when this
        activation code will expire. This should not generally be set (it's
        primarily used for testing), and is only applied if it's earlier than
        the signed expiration time in `activation_code`.
        """
        return self._req(Service.ENTERPRISE, "Activate", activation_code=activation_code, expires=expires).info

    def get_enterprise_state(self):
        """
        Gets the current enterprise state of the cluster. Returns a
        `GetEnterpriseResponse` object.
        """
        return self._req(Service.ENTERPRISE, "GetState")

    def deactivate_enterprise(self):
        """Deactivates enterprise."""
        return self._req(Service.ENTERPRISE, "Deactivate")
