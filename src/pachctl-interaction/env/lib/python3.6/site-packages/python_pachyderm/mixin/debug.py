from python_pachyderm.service import Service


class DebugMixin:
    def dump(self, recursed=None):
        """
        Gets a debug dump. Yields byte arrays.

        Params:

        * `recursed`: An optional bool.
        """
        res = self._req(Service.DEBUG, "Dump", recursed=recursed)
        for item in res:
            yield item.value

    def profile_cpu(self, duration):
        """
        Gets a CPU profile. Yields byte arrays.

        Params:

        * `duration`: A `Duration` object specifying how long to run the CPU
        profiler.
        """
        res = self._req(Service.DEBUG, "Profile", profile="cpu", duration=duration)
        for item in res:
            yield item.value

    def binary(self):
        """
        Gets the pachd binary. Yields byte arrays.
        """
        res = self._req(Service.DEBUG, "Binary")
        for item in res:
            yield item.value
