import random
import math

class Track:
    def __init__(self, track_layout):
        """
        Initialize track with layout and physical coefficients.
        
        Args:
            layout (list): List of track segments with length, gradient, and curve_radius
        """
        
        self.track_layout = track_layout
        self.current_segment_index = 0
        self.segment_position = 0.0
        
        # Track coefficients
        self.friction_coeff = 0.6
        self.curve_resistance_coeff = 0.6
        
        # Wind parameters
        self.wind_base_speed = 0.0  # m/s
        self.wind_direction = 0.0   # degrees (0 = headwind, 180 = tailwind)
        self.wind_variability = 0.0 # standard deviation for wind speed variation
        self.wind_gust_factor = 1.0 # multiplier for wind gusts
        
        # Pre-compute cumulative distances for efficient search
        self._cumulative_distances = self._compute_cumulative_distances()
        self.total_length = self._cumulative_distances[-1] if self._cumulative_distances else 0.0
       
    def _compute_cumulative_distances(self):
        """Pre-compute cumulative distances for efficient binary search."""
        cumulative = []
        total = 0.0
        for segment in self.track_layout:
            total += segment['length']
            cumulative.append(total)
        return cumulative
    
    def _binary_search_segment(self, position):
        """
        Use binary search to find segment index for given position.
        
        Args:
            position (float): Absolute position along track
            
        Returns:
            int: Segment index, or len(layout) if beyond track
        """
        if not self._cumulative_distances or position < 0:
            return 0
        
        if position >= self.total_length:
            return len(self.track_layout)
        
        left, right = 0, len(self._cumulative_distances) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if mid == 0:
                if position <= self._cumulative_distances[0]:
                    return 0
                left = mid + 1
            else:
                if (self._cumulative_distances[mid - 1] < position <= 
                    self._cumulative_distances[mid]):
                    return mid
                elif position <= self._cumulative_distances[mid - 1]:
                    right = mid - 1
                else:
                    left = mid + 1
        
        return len(self.track_layout)
    
    def reset(self):
        """Reset track position to start."""
        self.current_segment_index = 0
        self.segment_position = 0.0
    
    def set_wind_conditions(self, base_speed, direction=0.0, variability=0.0, gust_factor=1.0):
        """
        Set wind conditions for the track.
        
        Args:
            base_speed (float): Base wind speed in m/s
            direction (float): Wind direction in degrees (0=headwind, 180=tailwind)
            variability (float): Standard deviation for wind speed variation
            gust_factor (float): Multiplier for wind gusts
        """
        self.wind_base_speed = base_speed
        self.wind_direction = direction
        self.wind_variability = variability
        self.wind_gust_factor = gust_factor
    
    def get_wind_force(self, train_velocity, train_frontal_area, train_cd):
        """
        Calculate wind force acting on the train.
        
        Args:
            train_velocity (float): Train velocity in m/s
            train_frontal_area (float): Train frontal area in m²
            train_cd (float): Train drag coefficient
            
        Returns:
            float: Wind force in N (positive = resistance, negative = assistance)
        """
        # Generate wind speed with variability and gusts
        wind_variation = random.gauss(0, self.wind_variability) if self.wind_variability > 0 else 0
        gust_multiplier = random.uniform(1.0, self.wind_gust_factor)
        actual_wind_speed = (self.wind_base_speed + wind_variation) * gust_multiplier
        
        # Convert wind direction to relative wind velocity
        wind_direction_rad = math.radians(self.wind_direction)
        relative_wind_velocity = actual_wind_speed * math.cos(wind_direction_rad)
        
        # Calculate relative air velocity (train velocity + headwind component)
        air_velocity = train_velocity + relative_wind_velocity
        
        # Air density
        rho = 1.225  # kg/m³
        
        # Wind force calculation (drag equation)
        wind_force = 0.5 * rho * train_frontal_area * train_cd * air_velocity * abs(air_velocity)
        
        return wind_force
    
    def update_position(self, absolute_position):
        """
        Update track position using efficient binary search.
        
        Args:
            absolute_position (float): Absolute position along track
            
        Returns:
            bool: True if still on track, False if end reached
        """
        # Use binary search for efficiency
        segment_idx = self._binary_search_segment(absolute_position)
        
        if segment_idx >= len(self.track_layout):
            self.current_segment_index = len(self.track_layout)
            return False
        
        self.current_segment_index = segment_idx
        
        # Calculate position within current segment
        if segment_idx == 0:
            self.segment_position = absolute_position
        else:
            self.segment_position = absolute_position - self._cumulative_distances[segment_idx - 1]
        
        return True
    
    def get_current_properties(self, train_position=None):

        if train_position is not None:
            if not self.update_position(train_position):
                return None
        
        if self.current_segment_index >= len(self.track_layout):
            return None
            
        segment = self.track_layout[self.current_segment_index]
        return {
            'gradient': segment['gradient'],
            'curve_radius': segment['curve_radius'],
            'friction_coeff': self.friction_coeff,
            'curve_resistance_coeff': self.curve_resistance_coeff,
            'wind_speed': self.wind_base_speed,
            'wind_direction': self.wind_direction
        }
    
    def get_gradient(self):
        """Get current gradient percentage."""
        props = self.get_current_properties()
        return props['gradient'] if props else 0.0
    
    def get_curve_radius(self):
        """Get current curve radius (None for straight track)."""
        props = self.get_current_properties()
        return props['curve_radius'] if props else None
    
    def is_end_of_track(self):
        """Check if at end of track."""
        return self.current_segment_index >= len(self.track_layout)

default_track_layout = [
    {"length": 5000, "gradient": 0.0, "curve_radius": 100},
    {"length": 2000, "gradient": 1.5, "curve_radius": 500.0},
    {"length": 3000, "gradient": -1.0, "curve_radius": None},
    {"length": 1000, "gradient": 0.0, "curve_radius": 300.0},
]