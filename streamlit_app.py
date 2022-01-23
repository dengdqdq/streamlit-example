from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

# Welcome to Streamlit !

"""
# 广州城市理工学院
[streamlit例子](https://share.streamlit.io/dengdqdq/streamlit-example)

"""
with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
  

st.latex(r'''\Huge 
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')


st.latex(r'''\huge 
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')


st.latex(r'''\Large 
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

st.latex(r'''\large 
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

st.balloons()


"""
- OpenCV绘图介绍
- 在OpenCV中绘制基本形状，线、矩形和圆
- 基本形状(2)-剪贴线和带箭头的线、椭圆和折线
- 绘制文本
- 鼠标事件动态绘图
- 高级绘图

```python
cv2.line(image, (0, 0), (400, 400), colors['green'], 3)
cv2.line(image, (0, 400), (400, 0), colors['blue'], 10)
cv2.line(image, (200, 0), (200, 400), colors['red'], 3)
cv2.line(image, (0, 200), (400, 200), colors['yellow'], 10)
```
微分方程的解可能不是幂级数$\sum_{n=0}^\infty a_nx^n$,而是以下的情况


（a） 例如，包含x的负幂
$$
y=\\frac{cosx}{x^2}=\\frac{1}{x^2}-\\frac{1}{2!}+\\frac{x^2}{4!}-...
$$
（b) 例如，x的分数次方作为因数
$$
y=\sqrt{x}\,sin x = x^{1/2}(x-\\frac{x^3}{3!}+...) 
$$
这两种情况及其他,参见第21节,包含一种级数形式
$$
\\begin{aligned}
y= x^s\sum_{n=0}^\infty a_n x^n =\sum_{n=0}^\infty a_n x^{n+s} \\tag {11.1}
\\end{aligned}
$$


其中s为适合问题的量，可以是正数，负数，也可以是分数，甚至可以是复数，不过现在不考虑复数情况。$a_0x^s$是级数第一项，设$a_0$不为零。级数(11.1)称为广义幂级数。我们将考虑一些微分方程，这些方程可以假定（11.1）形式的级数解求解。这种微分方程解法称弗罗比尼乌斯法。



例1. 为了说明这种方法，解方程
$$
\\begin{aligned}
\\tag{11.2}
x^2y''+ 4xy'+ (x^2 + 2)y = 0
\\end{aligned}
$$

从（11.1）有

$$
\\begin{aligned}
y &= a_0x^s + a_1x^{s+1} + a_2x^{s+2} + ... =\sum_{n=0}^\infty  \\\\
y'&= sa_0x^{s−1} + (s + 1)a_1x^s + (s + 2)a_2x^{s+1} + ...   \\\\   
&=\sum_{n=0}^\infty (n+s)a_nx^{n+s-1}  \\\\
y''&=s(s−1)a_0x^{s−2} + (s+1)sa_1x^{s−1}+(s+2)(s+1)a_2x^s + ...  \\\\
&=\sum_{n=0}^\infty (n+s)(n+s-1)a_n x^{n+s-2} \\\\
\\tag{11.3}
\\end{aligned}
$$

将(11.3)代入(11.2)，对x幂列表。就像解勒让德方程一样

|           | $x^s$  |  $x^{s+1}$  |   $x^{s+2}$ |   ... $x^{s+n}$ |                 | 
| --------- | :----: | ----------: | ----------: | ----------: | :-----------------: |
|$x^2y''$|$s(s−1)a_0$|$(s+1)sa_1$|$(s+2)(s+1)a_2$|$(n+s)(n+s−1)a_n$|
|$4xy'$|$4sa_0$|$4(s + 1)a_1$|$4(s + 2)a_2$|$4(n + s)a_n$|
|$x^2y$|||$a_0$|$a_{n−2}$|
|$2y$|$2a_0$|$2a_1$|$2a_2$|$2a_n$|

x幂次的所有系数必为0，从$x^s$的系数，得$(s^2 + 3 s + 2) a_0 = 0$,由假设$a_0\\neq 0$，有
$$
s^2 + 3s + 2 = 0 \\tag{11.4}
$$
此s方程称指示方程，解得
$$s = −2,	s = −1.$$

求s =- 2和s = -1时的两个独立解，这两个独立解的线性组合即为方程的通解。就像Asinx +Bcosx是y’’ + y = 0的通解一样。




"""
st.markdown('# 微积分')
st.markdown('微积分由三个部分组成，即微分、积分以及联系微分﹑积分的微积分基本定理。')
st.markdown('## 一维微积分')
txt=r'''
若y=f(x)为定义在区间(a，b)上的一个函数﹐$\frac{x+y}{y+z}$，如果$\lim_{h\to\infty}$,$\frac{f(x+h)-f(h)}{h}$
在(a,b)中的一点x存在,则称f(x)在这点可微，记这极限值为或df/dx 或f'(x)，称df=f(x)dx为f(x)在x点的微分。
如果在(a ,b)上每一点都可微，则称函数在(a,b)上可微。
'''
st.markdown(txt)


st.line_chart({"data": [1, 5, 2, 6, 2, 1]})

with st.expander("展开"):
     st.latex(r'''\huge 
          a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
          \sum_{k=0}^{n-1} ar^k =
          a \left(\frac{1-r^{n}}{1-r}\right)
          ''')


     st.latex(r'''\Large 
          a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
          \sum_{k=0}^{n-1} ar^k =
          a \left(\frac{1-r^{n}}{1-r}\right)
          ''')

     st.latex(r'''\large 
          a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
          \sum_{k=0}^{n-1} ar^k =
          a \left(\frac{1-r^{n}}{1-r}\right)
          ''')

     st.image("https://static.streamlit.io/examples/dice.jpg")
   
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,2*np.pi,32)
fig = plt.figure()
plt.plot(x, np.sin(x),x,np.cos(x))
st.pyplot(fig)

st.line_chart({"data": [1, 5, 2, 6, 2, 1]})

with st.expander("展开"):
     st.latex(r'''\huge 
          a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
          \sum_{k=0}^{n-1} ar^k =
          a \left(\frac{1-r^{n}}{1-r}\right)
          ''')


     st.latex(r'''\Large 
          a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
          \sum_{k=0}^{n-1} ar^k =
          a \left(\frac{1-r^{n}}{1-r}\right)
          ''')

     st.latex(r'''\large 
          a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
          \sum_{k=0}^{n-1} ar^k =
          a \left(\frac{1-r^{n}}{1-r}\right)
          ''')

     st.image("https://static.streamlit.io/examples/dice.jpg")



f = lambda x : 1/(1+x**2)
a = 0; b = 5; N = 10
n = 10 # Use n*N+1 points to plot the function smoothly

x = np.linspace(a,b,N+1)
y = f(x)

X = np.linspace(a,b,n*N+1)
Y = f(X)

fig = plt.figure()
st.markdown('黎曼和，左')
plt.plot(X,Y,'b')
x_left = x[:-1] # Left endpoints
y_left = y[:-1]
plt.plot(x_left,y_left,'b.',markersize=10)
plt.bar(x_left,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='b')

st.pyplot(fig)
fig = plt.figure()
st.markdown('黎曼和，中')
plt.plot(X,Y,'b')
x_mid = (x[:-1] + x[1:])/2 # Midpoints
y_mid = f(x_mid)
plt.plot(x_mid,y_mid,'b.',markersize=10)
plt.bar(x_mid,y_mid,width=(b-a)/N,alpha=0.2,edgecolor='b')
st.pyplot(fig)


fig = plt.figure()
st.markdown('黎曼和，右')
plt.plot(X,Y,'b')
x_right = x[1:] # Left endpoints
y_right = y[1:]
plt.plot(x_right,y_right,'b.',markersize=10)
plt.bar(x_right,y_right,width=-(b-a)/N,alpha=0.2,align='edge',edgecolor='b')
st.pyplot(fig)
