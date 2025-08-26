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
        self.throttle_pid = PIDController(kp, ki, kd, output_limits=(0.0, 1.0))
        self.brake_pid = PIDController(kp, ki, kd, output_limits=(0.0, 1.0))
        
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
        speed_error = setpoint - current_speed
        
        if speed_error > 0:
            # Need to accelerate - use throttle
            throttle = self.throttle_pid.update(setpoint, current_speed, current_time)
            brake = 0.0
            # Reset brake PID to prevent windup
            self.brake_pid.reset()
        else:
            # Need to decelerate - use brake
            brake = self.brake_pid.update(current_speed, setpoint, current_time)
            throttle = 0.0
            # Reset throttle PID to prevent windup
            self.throttle_pid.reset()
        
        return throttle, brake
    
    def reset(self):
        """Reset controller state."""
        self.throttle_pid.reset()
        self.brake_pid.reset()


def main():
    """Test the PID speed controller with different scenarios."""
    
    # Test 1: Step response
    print("Testing PID Speed Controller")
    
    # Create controller
    controller = SpeedController(kp=0.8, ki=0.2, kd=0.1)
    
    # Scenario 1: Constant speed setpoint
    print("\n--- Test 1: Constant Speed (20 m/s) ---")
    controller.set_constant_speed(20.0)
    
    # Simulate simple dynamics
    dt = 0.1
    time_steps = 200
    
    # History arrays
    times = []
    speeds = []
    setpoints = []
    throttles = []
    brakes = []
    
    # Initial conditions
    current_speed = 0.0
    current_time = 0.0
    
    for step in range(time_steps):
        # Get control outputs
        throttle, brake = controller.update(current_speed, current_time)
        setpoint = controller.get_setpoint(current_time)
        
        # Simple vehicle dynamics (acceleration proportional to throttle-brake)
        acceleration = (throttle - brake) * 5.0  # max 5 m/s² acceleration
        current_speed += acceleration * dt
        current_speed = max(0.0, current_speed)  # No negative speeds
        
        # Store data
        times.append(current_time)
        speeds.append(current_speed)
        setpoints.append(setpoint)
        throttles.append(throttle)
        brakes.append(brake)
        
        current_time += dt
    
    # Scenario 2: Speed profile
    print("\n--- Test 2: Speed Profile ---")
    controller.reset()
    
    # Define speed profile: accelerate, cruise, decelerate
    profile_times = [0, 5, 15, 20]
    profile_speeds = [0, 25, 25, 10]
    controller.set_speed_profile(profile_times, profile_speeds)
    
    # Reset simulation
    current_speed = 0.0
    current_time = 0.0
    
    times2 = []
    speeds2 = []
    setpoints2 = []
    throttles2 = []
    brakes2 = []
    
    for step in range(time_steps):
        throttle, brake = controller.update(current_speed, current_time)
        setpoint = controller.get_setpoint(current_time)
        
        # Vehicle dynamics
        acceleration = (throttle - brake) * 5.0
        current_speed += acceleration * dt
        current_speed = max(0.0, current_speed)
        
        times2.append(current_time)
        speeds2.append(current_speed)
        setpoints2.append(setpoint)
        throttles2.append(throttle)
        brakes2.append(brake)
        
        current_time += dt
    
    # Plot results
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PID Speed Controller Test Results', fontsize=16, color='white')
    
    # Test 1 plots
    ax1.plot(times, setpoints, 'r--', label='Setpoint', linewidth=2)
    ax1.plot(times, speeds, 'cyan', label='Actual Speed', linewidth=2)
    ax1.set_title('Test 1: Constant Speed Control', color='white')
    ax1.set_xlabel('Time (s)', color='white')
    ax1.set_ylabel('Speed (m/s)', color='white')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(colors='white')
    
    ax2.plot(times, throttles, 'lime', label='Throttle', linewidth=2)
    ax2.plot(times, brakes, 'orange', label='Brake', linewidth=2)
    ax2.set_title('Test 1: Control Outputs', color='white')
    ax2.set_xlabel('Time (s)', color='white')
    ax2.set_ylabel('Control Output', color='white')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.tick_params(colors='white')
    
    # Test 2 plots
    ax3.plot(times2, setpoints2, 'r--', label='Setpoint', linewidth=2)
    ax3.plot(times2, speeds2, 'cyan', label='Actual Speed', linewidth=2)
    ax3.set_title('Test 2: Speed Profile Following', color='white')
    ax3.set_xlabel('Time (s)', color='white')
    ax3.set_ylabel('Speed (m/s)', color='white')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.tick_params(colors='white')
    
    ax4.plot(times2, throttles2, 'lime', label='Throttle', linewidth=2)
    ax4.plot(times2, brakes2, 'orange', label='Brake', linewidth=2)
    ax4.set_title('Test 2: Control Outputs', color='white')
    ax4.set_xlabel('Time (s)', color='white')
    ax4.set_ylabel('Control Output', color='white')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print(f"\nTest 1 - Final speed: {speeds[-1]:.2f} m/s (target: 20.0 m/s)")
    print(f"Test 1 - Settling time: ~{len([s for s in speeds if abs(s-20) > 1.0]) * dt:.1f}s")
    
    print(f"\nTest 2 - Final speed: {speeds2[-1]:.2f} m/s (target: {setpoints2[-1]:.1f} m/s)")
    
    return controller


if __name__ == "__main__":
    main()
