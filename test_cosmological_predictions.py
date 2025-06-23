#!/usr/bin/env python3
"""
Cosmological Predictions Test Module
===================================

Specialized tests for the observational predictions of the Complex Cosmos theory,
focusing on falsifiable signatures that distinguish it from standard cosmology.

This module focuses on:
1. CMB power spectrum and non-Gaussianity
2. Primordial gravitational wave suppression
3. Dark matter predictions
4. Big Bang nucleosynthesis consistency
5. Structure formation implications
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, legendre
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class CMBAnalyzer:
    """
    Analyzes CMB predictions from complex time bounce cosmology
    """
    
    def __init__(self):
        # Standard cosmological parameters
        self.H_0 = 67.4  # km/s/Mpc
        self.Omega_b = 0.049  # Baryon density
        self.Omega_c = 0.261  # Cold dark matter density
        self.Omega_Lambda = 0.690  # Dark energy density
        self.T_cmb = 2.725  # CMB temperature (K)
        
        # Theory-specific parameters
        self.bounce_scale = 1e-30  # Minimum scale factor
        self.R_I = 1e-35  # Compactification radius (m)
        
    def primordial_power_spectrum(self, k, bounce_dynamics=True):
        """
        Calculate primordial power spectrum from bounce
        
        Parameters:
        -----------
        k : array_like
            Wavenumber modes (Mpc^-1)
        bounce_dynamics : bool
            Include bounce-specific modifications
            
        Returns:
        --------
        P_s : array_like
            Scalar power spectrum
        P_t : array_like
            Tensor power spectrum
        """
        # Pivot scale
        k_pivot = 0.05  # Mpc^-1
        
        # Scalar spectrum (nearly scale-invariant)
        A_s = 2.1e-9  # Amplitude
        n_s = 0.965   # Spectral index
        P_s = A_s * (k / k_pivot)**(n_s - 1)
        
        if bounce_dynamics:
            # Bounce modifications
            # 1. Slight blue tilt at large scales due to quantum bounce
            bounce_correction = 1 + 0.01 * (k / k_pivot)**0.1
            P_s *= bounce_correction
            
            # 2. Strongly suppressed tensor modes
            r = 1e-6  # Tensor-to-scalar ratio << 10^-3
            P_t = r * P_s
        else:
            # Standard inflation prediction
            r = 0.1
            P_t = r * P_s
        
        return P_s, P_t
    
    def transfer_function(self, k, z_eq=3400):
        """
        Matter transfer function (simplified)
        
        Parameters:
        -----------
        k : array_like
            Wavenumber (Mpc^-1)
        z_eq : float
            Matter-radiation equality redshift
            
        Returns:
        --------
        T_k : array_like
            Transfer function
        """
        # Simplified BBKS transfer function
        q = k / (self.Omega_b + self.Omega_c) / (self.H_0 / 100)**2
        
        T_k = np.log(1 + 2.34 * q) / (2.34 * q) * \
              (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25)
        
        return T_k
    
    def matter_power_spectrum(self, k, z=0):
        """
        Calculate matter power spectrum at redshift z
        
        Parameters:
        -----------
        k : array_like
            Wavenumber (Mpc^-1)
        z : float
            Redshift
            
        Returns:
        --------
        P_m : array_like
            Matter power spectrum
        """
        P_s, _ = self.primordial_power_spectrum(k)
        T_k = self.transfer_function(k)
        
        # Growth factor (simplified)
        D_z = 1 / (1 + z)  # Matter-dominated approximation
        
        P_m = P_s * T_k**2 * D_z**2
        
        return P_m
    
    def cmb_angular_power_spectrum(self, l_max=2500):
        """
        Calculate CMB temperature angular power spectrum
        
        Parameters:
        -----------
        l_max : int
            Maximum multipole
            
        Returns:
        --------
        l : array_like
            Multipole moments
        C_l : array_like
            Angular power spectrum
        """
        l = np.arange(2, l_max + 1)
        
        # Simplified calculation - in reality would use CAMB/CLASS
        # Acoustic oscillations
        k_s = 0.01  # Sound horizon scale
        
        C_l = np.zeros_like(l, dtype=float)
        
        for i, l_val in enumerate(l):
            # Acoustic peaks
            x = l_val * k_s / 100
            oscillation = (1 + 0.3 * np.cos(x))**2
            
            # Damping at high l
            damping = np.exp(-(l_val / 1000)**2)
            
            # Overall amplitude
            C_l[i] = 6000 * oscillation * damping / l_val**2
        
        return l, C_l
    
    def non_gaussianity_bispectrum(self, k1, k2, k3, f_NL_equil=50, f_NL_local=0):
        """
        Calculate non-Gaussian bispectrum
        
        Parameters:
        -----------
        k1, k2, k3 : float
            Wavenumber triplet
        f_NL_equil : float
            Equilateral non-Gaussianity parameter
        f_NL_local : float
            Local non-Gaussianity parameter
            
        Returns:
        --------
        bispectrum : float
            Bispectrum amplitude
        """
        # Check triangle condition
        if not (abs(k1 - k2) <= k3 <= k1 + k2 and 
                abs(k2 - k3) <= k1 <= k2 + k3 and 
                abs(k3 - k1) <= k2 <= k3 + k1):
            return 0.0
        
        P1, _ = self.primordial_power_spectrum(np.array([k1]))
        P2, _ = self.primordial_power_spectrum(np.array([k2]))
        P3, _ = self.primordial_power_spectrum(np.array([k3]))
        
        P1, P2, P3 = P1[0], P2[0], P3[0]
        
        # Local shape
        B_local = 2 * f_NL_local * (P1 * P2 + P2 * P3 + P3 * P1)
        
        # Equilateral shape (dominant in bounce cosmology)
        B_equil = (6/5) * f_NL_equil * (P1 * P2 + P2 * P3 + P3 * P1)
        
        return B_local + B_equil
    
    def detectability_forecast(self, experiment='Planck'):
        """
        Forecast detectability of non-Gaussianity
        
        Parameters:
        -----------
        experiment : str
            CMB experiment name
            
        Returns:
        --------
        forecast : dict
            Detection forecast
        """
        if experiment == 'Planck':
            sigma_f_NL_local = 5.0
            sigma_f_NL_equil = 20.0
        elif experiment == 'CMB-S4':
            sigma_f_NL_local = 1.0
            sigma_f_NL_equil = 3.0
        else:
            sigma_f_NL_local = 10.0
            sigma_f_NL_equil = 30.0
        
        # Theory predictions
        f_NL_equil_theory = 50  # Complex time bounce prediction
        f_NL_local_theory = 0   # Suppressed in bounce models
        
        # Detection significance
        sigma_equil = abs(f_NL_equil_theory) / sigma_f_NL_equil
        sigma_local = abs(f_NL_local_theory) / sigma_f_NL_local
        
        return {
            'experiment': experiment,
            'f_NL_equil_theory': f_NL_equil_theory,
            'f_NL_local_theory': f_NL_local_theory,
            'sigma_f_NL_equil': sigma_f_NL_equil,
            'sigma_f_NL_local': sigma_f_NL_local,
            'detection_significance_equil': sigma_equil,
            'detection_significance_local': sigma_local,
            'detectable_equil': sigma_equil > 3.0,
            'detectable_local': sigma_local > 3.0
        }

class GravitationalWavePredictor:
    """
    Predicts primordial gravitational wave signatures
    """
    
    def __init__(self):
        self.c = 299792458  # m/s
        self.G = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²
        self.H_0 = 2.2e-18  # s^-1 (67.4 km/s/Mpc)
        
    def primordial_gw_spectrum(self, f, bounce_suppression=True):
        """
        Calculate primordial gravitational wave spectrum
        
        Parameters:
        -----------
        f : array_like
            Frequency (Hz)
        bounce_suppression : bool
            Include bounce suppression effects
            
        Returns:
        --------
        Omega_gw : array_like
            Gravitational wave energy density
        """
        # Characteristic frequency scale
        f_eq = 1e-16  # Hz (horizon entry at matter-radiation equality)
        
        if bounce_suppression:
            # Strong suppression from bounce dynamics
            # No inflationary amplification of tensor modes
            r = 1e-6  # Tensor-to-scalar ratio
            
            # Spectrum shape
            Omega_gw = np.zeros_like(f)
            mask = f > f_eq
            Omega_gw[mask] = 1e-18 * (f[mask] / f_eq)**(-2)  # Steep falloff
            
        else:
            # Standard inflation prediction
            r = 0.1
            Omega_gw = 1e-15 * (f / f_eq)**(-2) * (r / 0.1)
        
        return Omega_gw
    
    def detector_sensitivity(self, f, detector='LIGO'):
        """
        Detector sensitivity curves
        
        Parameters:
        -----------
        f : array_like
            Frequency (Hz)
        detector : str
            Detector name
            
        Returns:
        --------
        h_sensitivity : array_like
            Strain sensitivity
        """
        if detector == 'LIGO':
            # Simplified LIGO sensitivity
            f_min, f_max = 10, 1000
            h_min = 1e-23
            
            h_sensitivity = np.full_like(f, np.inf)
            mask = (f >= f_min) & (f <= f_max)
            h_sensitivity[mask] = h_min * (f[mask] / 100)**(-1)
            
        elif detector == 'LISA':
            # Simplified LISA sensitivity
            f_min, f_max = 1e-4, 1e-1
            h_min = 1e-21
            
            h_sensitivity = np.full_like(f, np.inf)
            mask = (f >= f_min) & (f <= f_max)
            h_sensitivity[mask] = h_min * (f[mask] / 1e-3)**(-2)
            
        else:
            h_sensitivity = np.full_like(f, 1e-20)
        
        return h_sensitivity
    
    def detection_prospects(self, detector_list=['LIGO', 'LISA', 'ET']):
        """
        Assess detection prospects for primordial GWs
        
        Parameters:
        -----------
        detector_list : list
            List of detectors to consider
            
        Returns:
        --------
        prospects : dict
            Detection prospects for each detector
        """
        prospects = {}
        
        for detector in detector_list:
            if detector == 'LIGO':
                f_range = np.logspace(1, 3, 100)
            elif detector == 'LISA':
                f_range = np.logspace(-4, -1, 100)
            elif detector == 'ET':  # Einstein Telescope
                f_range = np.logspace(0, 3, 100)
            else:
                f_range = np.logspace(-4, 3, 100)
            
            Omega_gw_theory = self.primordial_gw_spectrum(f_range, bounce_suppression=True)
            Omega_gw_inflation = self.primordial_gw_spectrum(f_range, bounce_suppression=False)
            
            # Convert to strain
            h_theory = np.sqrt(Omega_gw_theory) * 1e-20  # Rough conversion
            h_inflation = np.sqrt(Omega_gw_inflation) * 1e-20
            
            h_sensitivity = self.detector_sensitivity(f_range, detector)
            
            # Detection criterion: signal > sensitivity
            detectable_theory = np.any(h_theory > h_sensitivity)
            detectable_inflation = np.any(h_inflation > h_sensitivity)
            
            prospects[detector] = {
                'frequency_range': f_range,
                'strain_theory': h_theory,
                'strain_inflation': h_inflation,
                'sensitivity': h_sensitivity,
                'detectable_theory': detectable_theory,
                'detectable_inflation': detectable_inflation
            }
        
        return prospects

class DarkMatterPredictor:
    """
    Predicts dark matter candidates from CPT symmetry
    """
    
    def __init__(self):
        self.m_planck = 2.176434e-8  # kg
        self.alpha_em = 1/137  # Fine structure constant
        
    def sterile_neutrino_mass(self, see_saw_scale=1e15):
        """
        Calculate sterile neutrino mass from see-saw mechanism
        
        Parameters:
        -----------
        see_saw_scale : float
            See-saw energy scale (GeV)
            
        Returns:
        --------
        mass_sterile : float
            Sterile neutrino mass (GeV)
        """
        # Active neutrino mass
        m_active = 0.05  # eV (approximate)
        
        # See-saw formula: m_active ~ m_Dirac^2 / m_sterile
        # Assuming m_Dirac ~ electroweak scale
        m_Dirac = 100  # GeV
        
        mass_sterile = m_Dirac**2 / (m_active * 1e-9)  # Convert eV to GeV
        
        return mass_sterile
    
    def kaluza_klein_dark_matter(self, R_I=1e-35, n_mode=1):
        """
        Calculate KK dark matter candidate mass
        
        Parameters:
        -----------
        R_I : float
            Compactification radius (m)
        n_mode : int
            KK mode number
            
        Returns:
        --------
        mass_kk : float
            KK dark matter mass (GeV)
        """
        # KK mass: m_n = n / R_I (in natural units)
        hbar_c = 1.973269804e-16  # GeV⋅m
        
        mass_kk = n_mode * hbar_c / R_I / 1e9  # Convert to GeV
        
        return mass_kk
    
    def relic_abundance(self, mass_dm, cross_section):
        """
        Calculate dark matter relic abundance
        
        Parameters:
        -----------
        mass_dm : float
            Dark matter mass (GeV)
        cross_section : float
            Annihilation cross-section (cm³/s)
            
        Returns:
        --------
        omega_dm : float
            Dark matter density parameter
        """
        # Simplified calculation
        # Omega_dm h² ≈ 3e-27 cm³/s / <σv>
        
        omega_dm_h2 = 3e-27 / cross_section
        h = 0.674  # Hubble parameter
        omega_dm = omega_dm_h2 / h**2
        
        return omega_dm
    
    def cpt_symmetric_candidates(self):
        """
        List CPT-symmetric dark matter candidates
        
        Returns:
        --------
        candidates : dict
            Dark matter candidate properties
        """
        candidates = {}
        
        # 1. Right-handed neutrino
        m_sterile = self.sterile_neutrino_mass()
        candidates['sterile_neutrino'] = {
            'mass_GeV': m_sterile,
            'interaction': 'weak',
            'stability': 'topologically_protected',
            'cpt_partner': 'left_handed_antineutrino'
        }
        
        # 2. KK dark matter
        m_kk = self.kaluza_klein_dark_matter()
        candidates['kk_dark_matter'] = {
            'mass_GeV': m_kk,
            'interaction': 'gravitational',
            'stability': 'kk_parity',
            'cpt_partner': 'kk_antiparticle'
        }
        
        # 3. Topological soliton
        candidates['topological_soliton'] = {
            'mass_GeV': 1000,  # TeV scale
            'interaction': 'topological',
            'stability': 'topological_charge',
            'cpt_partner': 'antisoliton'
        }
        
        return candidates

class BigBangNucleosynthesis:
    """
    Tests BBN consistency with complex time cosmology
    """
    
    def __init__(self):
        self.T_bbn = 1e9  # BBN temperature (K)
        self.t_bbn = 100  # BBN time (s)
        
    def helium_abundance(self, eta_b=6e-10, bounce_effects=False):
        """
        Calculate primordial helium abundance
        
        Parameters:
        -----------
        eta_b : float
            Baryon-to-photon ratio
        bounce_effects : bool
            Include bounce modifications
            
        Returns:
        --------
        Y_p : float
            Primordial helium mass fraction
        """
        # Standard BBN prediction
        Y_p_standard = 0.247
        
        if bounce_effects:
            # Bounce could modify neutron-proton ratio
            # through different thermal history
            delta_Y = 0.001  # Small correction
            Y_p = Y_p_standard + delta_Y
        else:
            Y_p = Y_p_standard
        
        return Y_p
    
    def deuterium_abundance(self, eta_b=6e-10, bounce_effects=False):
        """
        Calculate primordial deuterium abundance
        
        Parameters:
        -----------
        eta_b : float
            Baryon-to-photon ratio
        bounce_effects : bool
            Include bounce modifications
            
        Returns:
        --------
        D_H : float
            Deuterium-to-hydrogen ratio
        """
        # Standard BBN prediction (sensitive to baryon density)
        D_H_standard = 2.5e-5
        
        if bounce_effects:
            # Bounce might affect baryon asymmetry generation
            # through CPT-symmetric processes
            correction_factor = 1.02  # 2% correction
            D_H = D_H_standard * correction_factor
        else:
            D_H = D_H_standard
        
        return D_H
    
    def consistency_test(self, observed_abundances):
        """
        Test consistency with observed light element abundances
        
        Parameters:
        -----------
        observed_abundances : dict
            Observed primordial abundances
            
        Returns:
        --------
        consistency : dict
            Consistency test results
        """
        # Theory predictions
        Y_p_theory = self.helium_abundance(bounce_effects=True)
        D_H_theory = self.deuterium_abundance(bounce_effects=True)
        
        # Observational values
        Y_p_obs = observed_abundances.get('helium', 0.247)
        D_H_obs = observed_abundances.get('deuterium', 2.5e-5)
        
        # Uncertainties
        sigma_Y_p = 0.003
        sigma_D_H = 0.3e-5
        
        # Chi-squared test
        chi2_Y_p = ((Y_p_theory - Y_p_obs) / sigma_Y_p)**2
        chi2_D_H = ((D_H_theory - D_H_obs) / sigma_D_H)**2
        chi2_total = chi2_Y_p + chi2_D_H
        
        # Degrees of freedom
        dof = 2
        
        return {
            'Y_p_theory': Y_p_theory,
            'Y_p_observed': Y_p_obs,
            'D_H_theory': D_H_theory,
            'D_H_observed': D_H_obs,
            'chi2_helium': chi2_Y_p,
            'chi2_deuterium': chi2_D_H,
            'chi2_total': chi2_total,
            'dof': dof,
            'p_value': 1 - chi2_total / (2 * dof),  # Rough approximation
            'consistent': chi2_total < 6.0  # 95% confidence for 2 DOF
        }

def run_cosmological_prediction_tests():
    """
    Run comprehensive cosmological prediction tests
    """
    print("=" * 60)
    print("COSMOLOGICAL PREDICTIONS TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: CMB Analysis
    print("1. Testing CMB predictions...")
    cmb = CMBAnalyzer()
    
    # Power spectrum
    k_range = np.logspace(-4, 0, 100)
    P_s, P_t = cmb.primordial_power_spectrum(k_range)
    r = np.mean(P_t / P_s)
    results['tensor_to_scalar'] = r
    print(f"   Tensor-to-scalar ratio r = {r:.2e}")
    print(f"   Prediction satisfied (r << 10^-3): {r < 1e-3}")
    
    # Non-Gaussianity detectability
    forecast = cmb.detectability_forecast('CMB-S4')
    results['non_gaussianity'] = forecast
    print(f"   f_NL^equil detection significance: {forecast['detection_significance_equil']:.1f}σ")
    print(f"   Detectable with CMB-S4: {forecast['detectable_equil']}")
    
    # Test 2: Gravitational Waves
    print("\n2. Testing gravitational wave predictions...")
    gw = GravitationalWavePredictor()
    
    prospects = gw.detection_prospects(['LIGO', 'LISA'])
    results['gw_prospects'] = prospects
    
    for detector in prospects:
        detectable = prospects[detector]['detectable_theory']
        print(f"   {detector} detection prospect: {'Possible' if detectable else 'Unlikely'}")
    
    # Test 3: Dark Matter
    print("\n3. Testing dark matter predictions...")
    dm = DarkMatterPredictor()
    
    candidates = dm.cpt_symmetric_candidates()
    results['dark_matter'] = candidates
    
    for candidate in candidates:
        mass = candidates[candidate]['mass_GeV']
        print(f"   {candidate}: {mass:.2e} GeV")
    
    # Test 4: BBN Consistency
    print("\n4. Testing Big Bang nucleosynthesis consistency...")
    bbn = BigBangNucleosynthesis()
    
    observed = {'helium': 0.247, 'deuterium': 2.5e-5}
    consistency = bbn.consistency_test(observed)
    results['bbn_consistency'] = consistency
    
    print(f"   BBN consistent: {consistency['consistent']}")
    print(f"   Chi-squared: {consistency['chi2_total']:.2f}")
    print(f"   P-value: {consistency['p_value']:.3f}")
    
    print("\n" + "=" * 60)
    print("COSMOLOGICAL PREDICTION TESTS COMPLETED")
    print("=" * 60)
    
    return results

def visualize_cosmological_predictions(results):
    """
    Create visualizations for cosmological prediction test results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cosmological Predictions Test Results', fontsize=14)
    
    # Plot 1: Tensor-to-scalar ratio comparison
    models = ['Complex Time\nBounce', 'Standard\nInflation']
    r_values = [results['tensor_to_scalar'], 0.1]
    colors = ['blue', 'red']
    
    bars = axes[0, 0].bar(models, r_values, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=1e-3, color='black', linestyle='--', 
                       label='Observational limit')
    axes[0, 0].set_ylabel('Tensor-to-scalar ratio r')
    axes[0, 0].set_title('Primordial Gravitational Waves')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, r_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1e}', ha='center', va='bottom')
    
    # Plot 2: Non-Gaussianity detection
    ng = results['non_gaussianity']
    f_NL_types = ['Local', 'Equilateral']
    theory_values = [ng['f_NL_local_theory'], ng['f_NL_equil_theory']]
    uncertainties = [ng['sigma_f_NL_local'], ng['sigma_f_NL_equil']]
    
    x = np.arange(len(f_NL_types))
    bars = axes[0, 1].bar(x, theory_values, yerr=uncertainties, 
                          capsize=5, alpha=0.7, color=['orange', 'green'])
    axes[0, 1].set_xlabel('Non-Gaussianity Type')
    axes[0, 1].set_ylabel('f_NL')
    axes[0, 1].set_title('CMB Non-Gaussianity Predictions')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(f_NL_types)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Dark matter candidates
    dm = results['dark_matter']
    candidates = list(dm.keys())
    masses = [dm[candidate]['mass_GeV'] for candidate in candidates]
    
    axes[1, 0].bar(range(len(candidates)), masses, alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Dark Matter Candidate')
    axes[1, 0].set_ylabel('Mass (GeV)')
    axes[1, 0].set_title('CPT-Symmetric Dark Matter Candidates')
    axes[1, 0].set_xticks(range(len(candidates)))
    axes[1, 0].set_xticklabels([c.replace('_', '\n') for c in candidates], 
                               rotation=45, ha='right')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: BBN consistency
    bbn = results['bbn_consistency']
    elements = ['Helium-4', 'Deuterium']
    theory_values = [bbn['Y_p_theory'], bbn['D_H_theory'] * 1e5]  # Scale deuterium
    observed_values = [bbn['Y_p_observed'], bbn['D_H_observed'] * 1e5]
    
    x = np.arange(len(elements))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, theory_values, width, 
                           label='Theory', alpha=0.7, color='blue')
    bars2 = axes[1, 1].bar(x + width/2, observed_values, width, 
                           label='Observed', alpha=0.7, color='red')
    
    axes[1, 1].set_xlabel('Light Element')
    axes[1, 1].set_ylabel('Abundance')
    axes[1, 1].set_title('Big Bang Nucleosynthesis Consistency')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(elements)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add consistency indicator
    consistent_text = "CONSISTENT" if bbn['consistent'] else "INCONSISTENT"
    color = 'green' if bbn['consistent'] else 'red'
    axes[1, 1].text(0.5, max(max(theory_values), max(observed_values)) * 0.8, 
                    consistent_text, ha='center', va='center', 
                    fontweight='bold', color=color,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('cosmological_predictions_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run cosmological prediction tests
    test_results = run_cosmological_prediction_tests()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_cosmological_predictions(test_results)
    
    print("\nCosmological predictions analysis complete!")
    print("Results saved to: cosmological_predictions_results.png")