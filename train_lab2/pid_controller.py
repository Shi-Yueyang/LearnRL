import math
import matplotlib.pyplot as plt


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0)):
        """
        Initialize PID controller.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain  
            kd (float): Derivative gain
            output_limits (tuple): Min and max output values
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min, self.output_max = output_limits
        
        # Internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = 0.0
        self.first_call = True
        
    def reset(self):
        """Reset controller internal state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = 0.0
        self.first_call = True
    
    def update(self, setpoint, measured_value, current_time):
        """
        Update PID controller and get control output.
        
        Args:
            setpoint (float): Desired value
            measured_value (float): Current measured value
            current_time (float): Current time
            
        Returns:
            float: Control output
        """
        # Calculate error
        error = setpoint - measured_value
        
        if self.first_call:
            self.prev_time = current_time
            self.prev_error = error
            self.first_call = False
            return 0.0
        
        # Calculate time delta
        dt = current_time - self.prev_time
        if dt <= 0.0:
            return 0.0
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = max(self.output_min, min(self.output_max, output))
        
        # Store values for next iteration
        self.prev_error = error
        self.prev_time = current_time
        
        return output


class SpeedController:
    def __init__(self, kp=0.5, ki=0.1, kd=0.05):
        """
        Initialize speed controller with separate throttle and brake PIDs.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
        """
        self.action_pid = PIDController(kp, ki, kd, output_limits=(-1.0, 1.0))
        
        self.speed_profile = None
        self.profile_times = None
        self.current_setpoint = 0.0
        
    def set_speed_profile(self, times, speeds):
        """
        Set speed profile trajectory.
        
        Args:
            times (list): Time points
            speeds (list): Speed values at each time point
        """
        self.profile_times = times
        self.speed_profile = speeds
    
    def set_constant_speed(self, speed):
        """
        Set constant speed setpoint.
        
        Args:
            speed (float): Constant target speed
        """
        self.current_setpoint = speed
        self.speed_profile = None
        self.profile_times = None
    
    def get_setpoint(self, current_time):
        """
        Get speed setpoint at current time.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            float: Target speed
        """
        if self.speed_profile is None:
            return self.current_setpoint
        
        # Interpolate speed profile
        if current_time <= self.profile_times[0]:
            return self.speed_profile[0]
        elif current_time >= self.profile_times[-1]:
            return self.speed_profile[-1]
        else:
            # Linear interpolation
            for i in range(len(self.profile_times) - 1):
                if self.profile_times[i] <= current_time <= self.profile_times[i + 1]:
                    t1, t2 = self.profile_times[i], self.profile_times[i + 1]
                    v1, v2 = self.speed_profile[i], self.speed_profile[i + 1]
                    return v1 + (v2 - v1) * (current_time - t1) / (t2 - t1)
        
        return self.current_setpoint
    
    def update(self, current_speed, current_time):
        """
        Update speed controller and get throttle/brake outputs.
        
        Args:
            current_speed (float): Current measured speed
            current_time (float): Current time
            
        Returns:
            tuple: (throttle_output, brake_output)
        """
        setpoint = self.get_setpoint(current_time)
        action = self.action_pid.update(setpoint,current_speed, current_time)
        return action
    
    def reset(self):
        """Reset controller state."""
        self.throttle_pid.reset()
        self.brake_pid.reset()


def main():
    from track import Track
    from train2 import Train2, high_speed_train_params_test
    """Sets up and runs a simple train simulation."""
    # Define track layout
    track_layout = [
        {"length": 5000, "gradient": 0.0, "curve_radius": 100},
        {"length": 2000, "gradient": 1.5, "curve_radius": 500.0},
        {"length": 3000, "gradient": -1.0, "curve_radius": None},
        {"length": 1000, "gradient": 0.0, "curve_radius": 300.0},
    ]

    # Create train and track objects
    train = Train2(params=high_speed_train_params_test)  # 1000 tons
    track = Track(track_layout)
    controller = SpeedController(kp=0.9, ki=0.1, kd=0.1)
    controller.set_constant_speed(10.0)

    # Set wind conditions (example: 5 m/s headwind with 2 m/s variability and gusts up to 1.5x)
    track.set_wind_conditions(
        base_speed=5.0, direction=0.0, variability=2.0, gust_factor=1.5
    )

    # Simulation parameters
    dt = 0.1
    simulation_time_limit = 10
    time_history = []
    position_history = []
    velocity_history = []
    acceleration_history = []
    action_history = []
    setpoints = []
    time = train.time
    vel = train.velocity
    while train.time < simulation_time_limit and not track.is_end_of_track():
        # Get current track properties (auto-updates track position)

        track_props = track.get_current_properties(train.position)
        if not track_props:
            break

        action = controller.update(vel, time)
        action_history.append(action)

        # Update train dynamics
        
        time, pos, vel, acc = train.update_dynamics(
            dt, action, track
        )
        time_history.append(time)
        position_history.append(pos)
        velocity_history.append(vel)
        acceleration_history.append(acc)

        setpoint = controller.get_setpoint(time)
        setpoints.append(setpoint)
    # Plot results
    plt.style.use("dark_background")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 12))
    fig.suptitle("Train Dynamics Simulation Results", fontsize=16, color="white")

    # Position, Velocity, Acceleration plots
    plots = [
        (
            ax1,
            [(time_history, position_history, "cyan", "position")],
            "Position (m)",
        ),
        (
            ax2,
            [
                (time_history, velocity_history, "lime", "velocity"),
                (time_history, setpoints, "cyan", "setpoint"),
            ],
            "Velocity (m/s)",
        ),
        (
            ax3,
            [(time_history, acceleration_history, "yellow", "acceleration")],
            "Acceleration (m/s²)",
        ),
        (
            ax4,
            [
                (time_history, action_history, "orange", "action"),
            ],
            "Control Output",
        ),
    ]

    for ax, datas, ylabel in plots:
        for data in datas:
            ax.plot(data[0], data[1], color=data[2], label=data[3])
        ax.set_title(f"{ylabel.split()[0]} over Time", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(colors="white")
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
