# 1

Ftrack = k * action + b

action = 0
Ftrack = m*c1
b = m*c1

action = a_high
k* a_high + b = Ftrack_max

k = (Ftrack_max - m*c1)/a_high

Ftrack = (Ftrack_max - m*c1)/a_high * action + m*c1

utrack = Ftrack / Ftrack_max = (1 - m*c1/Ftrack_max) * action + m*c1/Ftrack_max

$$
u_{\text{track}} =  \left(1 - \frac{m c_1}{F_{\text{track\_max}}}\right) \cdot \text{a} + \frac{m c_1}{F_{\text{track\_max}}}
$$

Ftrack = 0
action = -b/k = -((m*c1)*a_high)/(Ftrack_max - m*c1)