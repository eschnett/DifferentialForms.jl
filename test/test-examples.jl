using DifferentialForms

# Create a 0-form (a scalar):
e = one(Form{3})
@test e[] == 1
@test collect(e) == [1]

# Create a 1-form (a vector) from a tuple:
v = Form{3,1}((1, 2, 3))
@test v[1] == 1
@test v[2] == 2
@test v[3] == 3
@test collect(v) == [1, 2, 3]

# Create a 2-form (an axial vector) from a tuple:
a = Form{3,2}((1, 2, 3))
@test a[1, 2] == 1
@test a[1, 3] == 2
@test a[2, 3] == 3
@test collect(a) == [1, 2, 3]

# Create a 3-form (a pseudoscalar):
p = Form{3,3}((1,))
@test p[1, 2, 3] == 1
@test p[end] == 1
@test collect(p) == [1]

# Add, subtract, scale
a2 = 2 * a
a3 = a + a2
@test a3 == 3 * a

# Exterior product (type \wedge<tab>)
# q = wedge(v, a)
q = v ∧ a
@test q == Form{3,3}((2,))

# Dot product (type \cdot<tab>)
# b = dot(v, v)
b = v ⋅ v
@test b == 14 * one(Form{3})
c = a ⋅ a
@test c == 14 * one(Form{3})

# Cross product (type \times<tab>)
# w = cross(a, a)
w = v × v
@test w == zero(Form{3,1})
