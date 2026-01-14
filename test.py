from lp import LP
from main import solve

with open('BT-Example-3.5-std.json', 'r') as f:
    lp = LP(f.read())

print("Problem:")
print(lp)

result = solve(lp)

print("Solution:")
print(f"Primal: {result['primal']}")
print(f"Dual: {result['dual']}")

obj_value = sum(lp.objective[i] * result['primal'][i] for i in range(len(result['primal'])))
print(f"Objective value: {obj_value}")

optimal_check = lp.primal_dual_are_optimal(result['primal'], result['dual'])
print(f" optimal? {optimal_check[0]} (gap: {optimal_check[1]:.1e})")
