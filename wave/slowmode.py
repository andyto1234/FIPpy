import numpy as np


def solve_dvz(w_A, dv_A, c_s, L_rho, V_A):
    """
    Solve Equation (28) for δv_z (vertical slow mode wave speed) given the parameters. Laming 2015
    we consider fundamental mode for the slow mode wave n = 1 


    Parameters (in cgs units):
        w_A   : Alfvén wave angular frequency [rad/s]
        dv_A  : Alfvén wave amplitude [cm/s]
        c_s   : sound speed [cm/s]
        L_rho : density scale height [cm]
        V_A   : Alfvén speed [cm/s]

    Returns:
        dvz_solution : The computed slow mode vertical velocity perturbation δv_z [cm/s] (float or complex).
                       This is the solution with the smallest absolute magnitude.
    """
    n = 1
    w_s = 2*w_A/n
    
    # Calculate the dimensionless ratio c_s^2/V_A^2
    cs_va_ratio_squared = (c_s**2)/(V_A**2)
    
    # Calculate the dimensionless term (1 - c_s^2/V_A^2)
    dimensionless_term = 1 - cs_va_ratio_squared
    
    # Calculate root_term with proper unit handling
    # dv_A^2 has units of (cm/s)^2
    # (L_rho^2)*(w_s^2)*((1-c_s^2/V_A^2)^2) has units of cm^2 * (rad/s)^2 * dimensionless = (cm*rad/s)^2
    # To make units compatible, we need to convert the second term to (cm/s)^2
    root_term = dv_A**2 + (L_rho * w_s * dimensionless_term)**2
    
    # Calculate second_term with proper units
    # L_rho * w_s * (1-c_s^2/V_A^2) has units of cm * rad/s * dimensionless = cm*rad/s
    second_term = L_rho * w_s * dimensionless_term
    
    # Calculate dv_z
    dv_z = (-1j/2) * (np.sqrt(root_term) - second_term)
    
    return dv_z

# def solve_dvz(ks, w, ws, cs, Lp, Lrho, gamma, g, kA, dvA):
#     """
#     Solve Equation (14) for δv_z (vertical slow mode wave speed) given the parameters.

#     Parameters (in cgs units):
#         ks    : wave number for the slow mode wave [cm⁻¹]
#         w     : angular frequency of the slow mode oscillation [rad/s]
#         ws    : characteristic angular frequency (≈ 2·w_A, the Alfvén wave frequency) [rad/s]
#         cs    : sound speed [cm/s]
#         Lp    : pressure scale height [cm]
#         Lrho  : density scale height [cm]
#         gamma : adiabatic index (dimensionless)
#         g     : gravitational acceleration [cm/s²]
#         kA    : wave number of the Alfvén wave [cm⁻¹]
#         dvA   : Alfvén wave amplitude [cm/s]

#     Returns:
#         dvz_solution : The computed slow mode vertical velocity perturbation δv_z [cm/s] (float or complex).
#                        This is the solution with the smallest absolute magnitude.
#     """

#     # All quantities are assumed to be in cgs units (cm, cm/s, rad/s, etc.).

#     # -------------------------------
#     # STEP 1: Define intermediate variables if needed.
#     # (e.g., 1j represents the imaginary unit for complex arithmetic)

#     # -------------------------------
#     # STEP 2: Compute the coefficients a_4, a_3, a_2, a_1, and a_0 of the quartic equation:
#     #   (δv_z)^4 [ ... ] + (δv_z)^3 [ ... ] + (δv_z)^2 [ ... ] +
#     #   (δv_z)   [ ... ] + [constant term] = 0
#     #
#     # These coefficients are derived from Equation (14) in Laming (2012).

#     a_4 = -ks ** 3 / ws
#     a_3 = -3*ks**2 + (cs**2/Lrho - cs**2/(gamma*Lp))*(1/Lp + 1j*ks)*(2*ks**2/ws**2 + ks/(1j*ws*Lrho))


#     a_2 = (-3*ks*ws + (cs**2/Lrho - cs**2/(gamma*Lp))*(1/Lp + 1j*ks)*(2*ks/ws + 1/(1j*ws*Lrho)) + 
#           ks**3*cs**2/ws - 1j*ks**2*cs**2/(ws*Lp) - ks*cs**2/(gamma*w*Lp**2)) + (-1j*ks**2*cs**2/(gamma*ws*Lp) - 1j*ks**2*g/ws - g*ks/(ws*Lp) + 
#           (2j*ks + 1/Lrho)*(1j*kA*dvA**2*ks/ws + dvA**2*ks/(2*w*Lrho) + dvA**2*1j*ks**2/(2*ws)))
    
#     a_1 = -ws**2 + ks**2*cs**2 + (-1j*ks*cs**2/Lp - cs**2/(gamma*Lp**2) - 1j*ks*cs**2/(gamma*Lp) - 1j*ks*g - g/Lrho + 
#           (3j*ks + 1/Lrho)*(1j*ks*dvA**2 + 1j*dvA**2*ks/(2*Lrho) - dvA**2*ks**2/2))
#     a_0 = -ws*ks*dvA**2

#     # -------------------------------
#     # STEP 3: Assemble the coefficients and solve the polynomial.
#     coeffs = [a_4, a_3, a_2, a_1, a_0]
#     roots = np.roots(coeffs)

#     # -------------------------------
#     # STEP 4: Select the root with the smallest absolute value.
#     abs_values = [abs(r) for r in roots]
#     min_index = np.argmin(abs_values)
#     dvz_solution = roots[min_index]

#     return dvz_solution
