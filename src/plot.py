import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math

def run(tf, x, u, m, s, z, v_data):
    t = np.linspace(0, tf, num=len(m.T))
    r = np.array(x[0:3, :])
    v = np.array(x[3:6, :])
    z = np.array(z)[0]
    s = np.array(s)[0]
    u = np.array(u)
    m = np.array(m)[0]
    g0 = 9.80665
    
    Th = [np.linalg.norm(u[:, i]) * m[i] for i in range(len(v.T))]
    Th_ = [(Th[i] + Th[i + 1]) / 2 for i in range(len(Th) - 1)] + [0]

    vnorm = [np.linalg.norm(vel) for vel in v.T]

    traj = go.Figure()
    traj.add_trace(go.Scatter3d(x=r[1, :], y=r[2, :], z=r[0, :], mode='lines', name='Flight Path'))

    r_ = np.linspace(0, max(max(r[1, :]), max(r[2, :])), 7)
    a_ = np.linspace(0, 2 * np.pi, 20)
    R, P = np.meshgrid(r_, a_)
    X, Y, Z = R * np.cos(P), R * np.sin(P), R * (np.tan(v_data['angle_gs']))
    xf = r[1, -1]
    yf = r[2, -1]
    zf = r[0, -1]
    Z = R * (np.tan(v_data['angle_gs'])) + zf
    X = X + xf
    Y = Y + yf

    traj.add_trace(go.Surface(x=X, y=Y, z=Z,
                               colorscale='YlGnBu_r', showscale=False,
                               name='Glide Slope Constraint'))

    for i in range(1, len(t), 2):
        scale_factor = 0.8*1e6
        thrust_magnitude = np.linalg.norm(u[:, i])
        thrust_percentage = thrust_magnitude / v_data['thrust_max']
        thrust_dir = u[:, i] / thrust_magnitude
        thrust_length = thrust_percentage * scale_factor
        traj.add_trace(go.Scatter3d(x=[r[1, i], r[1, i] + thrust_length * thrust_dir[1]],
                                     y=[r[2, i], r[2, i] + thrust_length * thrust_dir[2]],
                                     z=[r[0, i], r[0, i] + thrust_length * thrust_dir[0]],
                                     mode='lines',
                                     line=dict(color='red'),
                                     showlegend=False,
                                     legendgroup='Thrust Vector',
                                     name='Thrust Vector'))

    traj.add_trace(go.Scatter3d(x=[None], y=[None], z=[None],
                                 mode='lines',
                                 line=dict(color='red', width=2),
                                 legendgroup='Thrust Vector',
                                 name='Thrust Vector'))

    z0 = v_data['initial_state'][0]
    x0 = v_data['initial_state'][1]
    y0 = v_data['initial_state'][2]

    zp = v_data['landing_point'][0]
    xp = v_data['landing_point'][1]
    yp = v_data['landing_point'][2]

    traj.add_trace(go.Scatter3d(x=[x0], y=[y0], z=[z0],
                                  mode='markers', marker=dict(size=8, color='green'),
                                  name='Initial Position'))
    traj.add_trace(go.Scatter3d(x=[xp], y=[yp], z=[zp],
                                  mode='markers', marker=dict(size=8, color='red'),
                                  name='Planned Landing Point'))
    traj.add_trace(go.Scatter3d(x=[xf], y=[yf], z=[zf],
                                  mode='markers', marker=dict(size=8, color='blue'),
                                  name='Closet Landing Point'))

    traj.update_layout(
        scene=dict(aspectmode='data'),
        scene_xaxis_title=r'$x{1}$',
        scene_yaxis_title=r'$x{2}$',
        scene_zaxis_title=r'$x{0}$',
        legend=dict(y=0.05, x=0.05),
        title_text='Trajectory in 3D'
    )

    fig = make_subplots(rows=6, cols=1, 
                        subplot_titles=("Velocity Magnitude (m/s)", "Altitude (m)", "Mass (kg)",
                                        "Thrust (N)", "Thrust Angle", "Sigma Slack"))

    fig.add_trace(go.Scatter(x=t, y=vnorm, 
                             mode='lines', 
                             name='Velocity Magnitude (m/s)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(0, int(max(t))), y=np.full(int(max(t)), v_data['velocity_max']),
                              mode='lines', name='Maximum Velocity'), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=r[0, :], 
                             mode='lines', name='Altitude (m)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=m,
                             mode='lines', name='Mass (kg)'), row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=Th, 
                             mode='lines', name='Thrust (N)'), row=4, col=1)
    fig.add_trace(go.Scatter(x=np.arange(0, int(max(t))), y=np.full(int(max(t)), v_data['thrust_max']),
                              mode='lines', name='Maximum Thrust'), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=Th_, 
                             mode='lines', name='Modified Thrust (N)'), row=4, col=1)

    u_angle = [np.degrees(math.acos(min(1, ui[0] / np.linalg.norm(ui)))) for ui in u.T]
    fig.add_trace(go.Scatter(x=np.arange(0, int(max(t))), y=np.full(int(max(t)), np.degrees(v_data['angle_pt'])),
                              mode='lines', name='Desired Thrust Angle'), row=5, col=1)
    fig.add_trace(go.Scatter(x=t, y=u_angle, mode='lines', name='Thrust Angle'), row=5, col=1)

    alpha = 1 / (g0 * v_data['Isp'])
    z0_term = (v_data['mass_dry'] + v_data['mass_fuel']) - alpha * v_data['thrust_max']
    lim = []
    n = 0
    for t_ in t:
        if t_ > 0:
            try:
                v = v_data['thrust_max'] * v_data['throttle'][1] / \
                    (z0_term * t_) * (1 - (z[n] - np.log(z0_term * t_)))
            except ZeroDivisionError:
                v = 0
            lim.append(v)
        else:
            lim.append(0)
        n += 1

    fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name='Sigma Slack'), row=6, col=1)

    fig.update_layout(title_text='Rocket Data', showlegend=False)
    fig.update_layout(height=1200)

    traj.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        camera_projection=dict(type='orthographic'),
        xaxis=dict(gridcolor='grey', gridwidth=1),
        yaxis=dict(gridcolor='grey', gridwidth=1),
        zaxis=dict(gridcolor='grey', gridwidth=1),
    ))

    traj.show()
    fig.show()
