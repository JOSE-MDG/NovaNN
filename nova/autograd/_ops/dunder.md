📌 Operaciones aritméticas básicas

- `__add__(self, other)` → `a + b`
- `__sub__(self, other)` → `a - b`
- `__mul__(self, other)` → `a * b`
- `__matmul__(self, other)` → `a @ b`
- `__truediv__(self, other)` → `a / b`
- `__floordiv__(self, other)` → `a // b`
- `__mod__(self, other)` → `a % b`
- `__divmod__(self, other)` → `divmod(a, b)`
- `__pow__(self, other)` → `a ** b`
- `__lshift__(self, other)` → `a << b`
- `__rshift__(self, other)` → `a >> b`
- `__and__(self, other)` → `a & b`
- `__xor__(self, other)` → `a ^ b`
- `__or__(self, other)` → `a | b`

📌 Versiones "reflected" (cuando el operando izquierdo no soporta la operación)

- `__radd__`
- `__rsub__`
- `__rmul__`
- `__rmatmul__`
- `__rtruediv__`
- `__rfloordiv__`
- `__rmod__`
- `__rdivmod__`
- `__rpow__`
- `__rlshift__`
- `__rrshift__`
- `__rand__`
- `__rxor__`
- `__ror__`

📌 Versiones "in-place" (operadores como +=, *=, etc.)

- `__iadd__`
- `__isub__`
- `__imul__`
- `__imatmul__`
- `__itruediv__`
- `__ifloordiv__`
- `__imod__`
- `__ipow__`
- `__ilshift__`
- `__irshift__`
- `__iand__`
- `__ixor__`
- `__ior__`

📌 Operadores unarios

- `__neg__(self)` → `-a`
- `__pos__(self)` → `+a`
- `__abs__(self)` → `abs(a)`
- `__invert__(self)` → `~a`

Operador 	Método Dunder	Descripción
==	__eq__(self, other)	Comprueba si dos objetos son iguales.
!=	__ne__(self, other)	Comprueba si dos objetos son distintos.
<	__lt__(self, other)	Comprueba si es menor que (less than).
<=	__le__(self, other)	Comprueba si es menor o igual que (less or equal).
>	__gt__(self, other)	Comprueba si es mayor que (greater than).
>=	__ge__(self, other)	Comprueba si es mayor o igual que (greater or equal).