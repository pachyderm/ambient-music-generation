from contextlib import contextmanager

from python_pachyderm.proto.transaction import transaction_pb2 as transaction_proto
from python_pachyderm.service import Service


def transaction_from(transaction):
    if isinstance(transaction, transaction_proto.Transaction):
        return transaction
    else:
        return transaction_proto.Transaction(id=transaction)


class TransactionMixin:
    def batch_transaction(self, requests):
        """
        Executes a batch transaction.

        Params:

        * `requests`: A list of `TransactionRequest` objects.
        """
        return self._req(Service.TRANSACTION, "BatchTransaction", requests=requests)

    def start_transaction(self):
        """
        Starts a transaction.
        """
        return self._req(Service.TRANSACTION, "StartTransaction")

    def inspect_transaction(self, transaction):
        """
        Inspects a given transaction.

        Params:

        * `transaction`: A string or `Transaction` object.
        """
        return self._req(Service.TRANSACTION, "InspectTransaction", transaction=transaction_from(transaction))

    def delete_transaction(self, transaction):
        """
        Deletes a given transaction.

        Params:

        * `transaction`: A string or `Transaction` object.
        """
        return self._req(Service.TRANSACTION, "DeleteTransaction", transaction=transaction_from(transaction))

    def delete_all_transactions(self):
        """
        Deletes all transactions.
        """
        return self._req(Service.TRANSACTION, "DeleteAll")

    def list_transaction(self):
        """
        Lists transactions.
        """
        return self._req(Service.TRANSACTION, "ListTransaction").transaction_info

    def finish_transaction(self, transaction):
        """
        Finishes a given transaction.

        Params:

        * `transaction`: A string or `Transaction` object.
        """
        return self._req(Service.TRANSACTION, "FinishTransaction", transaction=transaction_from(transaction))

    @contextmanager
    def transaction(self):
        """
        A context manager for running operations within a transaction. When
        the context manager completes, the transaction will be deleted if an
        error occurred, or otherwise finished.
        """

        old_transaction_id = self.transaction_id
        transaction = self.start_transaction()
        self.transaction_id = transaction.id

        try:
            yield transaction
        except Exception:
            self.delete_transaction(transaction)
            raise
        else:
            self.finish_transaction(transaction)
        finally:
            self.transaction_id = old_transaction_id
