import catanatron
import catanatron.models.actions as actions_mod
from catanatron import Action as RootAction
from catanatron.models.actions import Action as ModelAction

print(f"RootAction: {RootAction}")
print(f"ModelAction: {ModelAction}")
print(f"Are they same? {RootAction is ModelAction}")

import inspect
print(f"RootAction defined in: {inspect.getfile(RootAction)}")
print(f"ModelAction defined in: {inspect.getfile(ModelAction)}")

# Check if they have 'before' attribute?
print(f"RootAction Dir: {dir(RootAction)}")
print(f"ModelAction Dir: {dir(ModelAction)}")
