from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit !
## 广州城市理工学院

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
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

# plt.show()
# plt.plot(x,np.exp(x))
# plt.show()

# x=np.linspace(-0.5*np.pi,0.5*np.pi,100)
# plt.plot(x,np.tan(x))
# plt.show()

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

fig=plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(X,Y,'b')
x_left = x[:-1] # Left endpoints
y_left = y[:-1]
plt.plot(x_left,y_left,'b.',markersize=10)
plt.bar(x_left,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='b')
plt.title('Left Riemann Sum, N = {}'.format(N))

plt.subplot(1,3,2)
plt.plot(X,Y,'b')
x_mid = (x[:-1] + x[1:])/2 # Midpoints
y_mid = f(x_mid)
plt.plot(x_mid,y_mid,'b.',markersize=10)
plt.bar(x_mid,y_mid,width=(b-a)/N,alpha=0.2,edgecolor='b')
plt.title('Midpoint Riemann Sum, N = {}'.format(N))    

plt.subplot(1,3,3)
plt.plot(X,Y,'b')
x_right = x[1:] # Left endpoints
y_right = y[1:]
plt.plot(x_right,y_right,'b.',markersize=10)
plt.bar(x_right,y_right,width=-(b-a)/N,alpha=0.2,align='edge',edgecolor='b')
plt.title('Right Riemann Sum, N = {}'.format(N))




st.markdown('黎曼和，左')
plt.plot(X,Y,'b')
x_left = x[:-1] # Left endpoints
y_left = y[:-1]
plt.plot(x_left,y_left,'b.',markersize=10)
plt.bar(x_left,y_left,width=(b-a)/N,alpha=0.2,align='edge',edgecolor='b')

st.markdown('黎曼和，中')
plt.plot(X,Y,'b')
x_mid = (x[:-1] + x[1:])/2 # Midpoints
y_mid = f(x_mid)
plt.plot(x_mid,y_mid,'b.',markersize=10)
plt.bar(x_mid,y_mid,width=(b-a)/N,alpha=0.2,edgecolor='b')

st.markdown('黎曼和，右')
plt.plot(X,Y,'b')
x_right = x[1:] # Left endpoints
y_right = y[1:]
plt.plot(x_right,y_right,'b.',markersize=10)
plt.bar(x_right,y_right,width=-(b-a)/N,alpha=0.2,align='edge',edgecolor='b')

st.pyplot(fig)
