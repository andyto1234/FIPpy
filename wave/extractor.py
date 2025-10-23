import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from astropy import units as u
from physics.speed import AlfvenSpeedCalculator
from wave.solver import WaveAnalyzer

@dataclass
class WaveComponents:
    """Data class to store wave components and derived quantities."""
    s_array: np.ndarray
    I_plus: np.ndarray
    I_minus: np.ndarray
    flux_right: np.ndarray
    flux_left: np.ndarray
    flux_difference: np.ndarray
    normalized_flux_difference: np.ndarray
    ponderomotive_acceleration: np.ndarray
    # wave_number: np.ndarray
    solution: Any  # Full ODE solution object

class WaveSolutionExtractor:
    """Class responsible for extracting and processing wave solution data."""
    
    def __init__(self, loop_properties, sol):
        """
        Initialize the extractor with loop properties.
        
        Args:
            loop_properties: Object containing loop physical properties
        """
        self.loop_properties = loop_properties
        # self.frequency = frequency
        self.light_speed_cgs = 2.99792458e+10 * u.cm / u.s
        self.sol = sol
        self.coordinate = self.sol.t * u.cm
        self.alfven_speed = WaveAnalyzer(self.loop_properties).VA(self.coordinate)

    def process_solution(self) -> WaveComponents:
        """
        Process the ODE solution and extract wave components.
        
        Args:
            sol: Solution object from solve_ivp
            s_array_unique: Array of unique position values
            
        Returns:
            WaveComponents: Dataclass containing all processed wave components
        """
        I_plus = self._extract_forward_wave(self.sol)
        I_minus = self._extract_backward_wave(self.sol)
        
        flux_right = self._calculate_energy_flux(
            I_plus
        )
        flux_left = self._calculate_energy_flux(
            I_minus
        )
        flux_difference = flux_right - flux_left
        normalized_flux_difference = flux_difference.value / self.loop_properties.B(self.coordinate).value
        ponderomotive_acceleration = self._calculate_ponderomotive_acceleration(
            I_plus, 
            I_minus
        )
        
        # wave_number = self._calculate_wave_number(self.coordinate)
        
        return WaveComponents(
            s_array=self.coordinate,
            I_plus=I_plus,
            I_minus=I_minus,
            flux_right=flux_right,
            flux_left=flux_left,
            flux_difference=flux_difference,
            normalized_flux_difference=normalized_flux_difference,
            ponderomotive_acceleration=ponderomotive_acceleration,
            # wave_number=wave_number,
            solution=self.sol
        )

    def _extract_forward_wave(self, sol: Any) -> np.ndarray:
        """Extract forward propagating wave component."""
        return (sol.y[0]*u.cm/u.s + 1j * sol.y[1]*u.cm/u.s)

    def _extract_backward_wave(self, sol: Any) -> np.ndarray:
        """Extract backward propagating wave component."""
        return (sol.y[2]*u.cm/u.s + 1j * sol.y[3]*u.cm/u.s)

    def _calculate_energy_flux(
        self, 
        I: np.ndarray, 
    ) -> np.ndarray:
        """Calculate wave energy density."""
        # print(self.loop_properties.rho(self.coordinate).unit)
        # print(self.alfven_speed.unit)
        # print(I.unit)
        return (0.25 * self.loop_properties.rho(self.coordinate) * (
            np.abs(I)**2 * self.alfven_speed)).to(u.erg/u.cm**2/u.s)
        

    def _calculate_ponderomotive_acceleration(
        self, 
        I_plus: np.ndarray, 
        I_minus: np.ndarray
    ) -> np.ndarray:
        """Calculate ponderomotive force."""
        delta_E_square = ((self.loop_properties.B(self.coordinate)**2 / 2) / self.light_speed_cgs**2) * (np.abs(I_plus)**2 + np.abs(I_minus)**2)
        delta_E_squared_over_B_squared = delta_E_square / (self.loop_properties.B(self.coordinate)**2)
        gradient = np.gradient(delta_E_squared_over_B_squared, self.coordinate)
        return 0.25 * gradient * self.light_speed_cgs**2

    # def _calculate_wave_number(self) -> np.ndarray:
    #     """Calculate wave number."""
    #     omega = self._frequency_to_angular_frequency()
    #     return omega / self.alfven_speed

    # def _frequency_to_angular_frequency(self) -> float:
    #     """Convert frequency to angular frequency."""
    #     return 2 * np.pi * self.frequency

    # def _calculate_alfven_speed(self) -> np.ndarray:
    #     """Calculate Alfvén speed at given positions."""
    #     return self.loop_properties.VA(self.coordinate)