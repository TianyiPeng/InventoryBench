""" Register VendingMachine environment """ 

from or_agent.envs.registration import register_with_versions
from or_agent.wrappers import ActionFormattingWrapper

# Vending Machine (2 players, simple)
from or_agent.envs.VendingMachine.wrapper import VendingMachineObservationWrapper
register_with_versions(
    id="VendingMachine-v0",
    entry_point="or_agent.envs.VendingMachine.env:VendingMachineEnv",
    wrappers={"default": [VendingMachineObservationWrapper, ActionFormattingWrapper], "-train": [VendingMachineObservationWrapper, ActionFormattingWrapper]},
)