import math
import matplotlib.pyplot as plt

class Train:
    def __init__(self, params):
        """
        Initializes the Train object with a given mass.
        
        Args:
            mass (float): The total mass of the train in kg.
        """
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.time = 0.0
        
        # Physical constants
        self.g = 9.81  # m/s^2
        self.rho = 1.225  # kg/m^3
        
        # Train-specific parameters
        self.m = params['mass']
        self.A = params['frontal_area']  # Frontal Area m^2
        self.Cd = params['drag_coefficient']  # Drag Coefficient
        self.c1 = params['c1']  # N/kg
        self.c2 = params['c2'] # Ns/kgm
        self.c3 = params['c3'] # Ns^2/kgm^2
        self.max_brake_force = params['max_brake_force']  # N
        self.max_tractive_force = params['max_tractive_force'] # N
        
    def calculate_tractive_effort(self, throttle_input):

        return throttle_input * self.max_tractive_force
        
    def calculate_resistance_forces(self, velocity, track_props, track):
        """Calculate total resistance forces acting on the train."""
        gradient_percent = track_props['gradient']
        curve_radius = track_props['curve_radius']
        curve_resistance_coeff = track_props['curve_resistance_coeff']

        # Rolling Resistance (Davis Equation)
        f_rolling = self.c1 + self.c2 * velocity + self.c3 * velocity**2
        
        # Aerodynamic Drag (base aerodynamic resistance without wind)
        f_drag = 0.5 * self.rho * self.A * self.Cd * velocity**2
        
        # Wind Force (environmental disturbance)
        f_wind = track.get_wind_force(velocity, self.A, self.Cd)
        
        # Gravitational Force
        theta = math.atan(gradient_percent / 100.0)
        f_gravity = self.m * self.g * math.sin(theta)
        
        # Curve Resistance
        f_curve = 0.0
        if curve_radius is not None and curve_radius > 0:
            f_curve = (self.m * self.g * curve_resistance_coeff) / curve_radius
        
        return f_rolling + f_drag + f_wind + f_gravity + f_curve

    def update_dynamics(self, dt, action, track):
        """Update train dynamics based on control inputs and track properties."""
        # Calculate forces
        if action>0:
            throttle_input = action
            brake_input = 0
        else:
            brake_input = -action
            throttle_input = 0
        track_props = track.get_current_properties(self.position)
        if track_props is None:
            return self.time, self.position, self.velocity, self.acceleration
        f_tractive = self.calculate_tractive_effort(throttle_input)
        f_resistance = self.calculate_resistance_forces(self.velocity, track_props, track)
        f_braking = brake_input * self.max_brake_force
        
        # Calculate net force and acceleration
        f_net = f_tractive - (f_resistance + f_braking)
        if self.velocity <= 0 and f_net <= 0:
            acceleration = 0.0
            self.velocity = 0.0
        else:
            acceleration = f_net / self.m
                
        # Update velocity and position using Euler integration
        self.velocity += acceleration * dt
        self.position += self.velocity * dt + 0.5 * acceleration * dt**2
        self.acceleration = acceleration
        self.time += dt

        return self.time, self.position, self.velocity, self.acceleration

high_speed_train_params_test = {
    'mass': 360_000,          # 10,000 metric tons (10^7 kg)
    'frontal_area': 20,         # 20 m^2
    'drag_coefficient': 0.9,    # High drag
    'c1': 2700,                  # Rolling resistance is significant
    'c2': 100,                 # Velocity-based resistance is low
    'c3': 7.1,            # Small coefficient for quadratic drag
    'max_brake_force': 1_000_000, # 1,000 kN (limited by momentum)
    'max_tractive_force': 25_000_000, # 1,500 kN (very high for starting)
}
    
def main():
    from track import Track
    
    """Sets up and runs a simple train simulation."""
    # Define track layout
    track_layout = [
        {"length": 5000, "gradient": 0.0, "curve_radius": 100},
        {"length": 2000, "gradient": 1.5, "curve_radius": 500.0},
        {"length": 3000, "gradient": -1.0, "curve_radius": None},
        {"length": 1000, "gradient": 0.0, "curve_radius": 300.0},
    ]
    
    # Create train and track objects

    train = Train(high_speed_train_params_test)
    track = Track(track_layout)
    
    # Set wind conditions (example: 5 m/s headwind with 2 m/s variability and gusts up to 1.5x)
    track.set_wind_conditions(base_speed=5.0, direction=0.0, variability=2.0, gust_factor=1.5)
    
    # Simulation parameters
    dt = 0.1
    simulation_time_limit = 1000.0
    time_history = []
    position_history = []
    velocity_history = []
    acceleration_history = []

    while train.time < simulation_time_limit and not track.is_end_of_track():
        # Get current track properties (auto-updates track position)

        # Simple control logic
        throttle = 1.0 if train.position <= 8000 else 0.0
        brake = 0.5 if train.position > 8000 else 0.0
        
        # Update train dynamics
        time, pos, vel, acc = train.update_dynamics(dt, throttle, brake, track)
        time_history.append(time)
        position_history.append(pos)
        velocity_history.append(vel)
        acceleration_history.append(acc)
        
    # Plot results
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Train Dynamics Simulation Results', fontsize=16, color='white')

    # Position, Velocity, Acceleration plots
    plots = [
        (ax1, time_history, position_history, 'Position (m)', 'cyan'),
        (ax2, time_history, velocity_history, 'Velocity (m/s)', 'lime'),
        (ax3, time_history, acceleration_history, 'Acceleration (m/s²)', 'yellow')
    ]
    
    for ax, x_data, y_data, ylabel, color in plots:
        ax.plot(x_data, y_data, color=color, label=ylabel.split()[0])
        ax.set_title(f'{ylabel.split()[0]} over Time', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(colors='white')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
