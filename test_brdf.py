import pytest

import numpy as np
from brdf import (EPSILON, grid_sample, projected_area, sphere_surface_patch,
                  visible_ndf, build_ndf_kernel, build_slope_kernel,
                  power_iteration, vndf_intp2sample, ndf_intp2sample,
                  normalize_slopes, normalize_2D, normalize_4D)

import mitsuba
# Set the any mitsuba variant
mitsuba.set_variant('gpu_spectral')

@pytest.mark.parametrize("n_theta", [128, 64])
@pytest.mark.parametrize("n_phi", [1, 64])
def test01_grid_sample(n_theta, n_phi):
    # Map samples to sphere
    theta_m, phi_m, omega_m = grid_sample(n_theta, n_phi)

    # Check if direction vectors are normalized
    assert((np.linalg.norm(omega_m, axis=1) - 1 < 1e-7).all())
    # Check grid dimensions
    N = n_theta * n_phi
    assert(theta_m.size == N)
    assert(phi_m.size == N)
    assert(omega_m.shape == (N, 3))


def test02_sphere_surface_patch():
    # Create input values
    r = 1
    dtheta = [0, np.pi/2]
    dphi = [0, np.pi]

    # Calsulate surface area of patch on sphere
    area = sphere_surface_patch(r, dtheta, dphi)

    error = np.abs(area - np.pi)
    assert(error < 1e-7)


@pytest.mark.parametrize("n_theta", [128])
@pytest.mark.parametrize("n_phi", [1])
@pytest.mark.parametrize("isotropic", [True])
def test03_sphere_integration(n_theta, n_phi, isotropic):
    # Create input values
    if isotropic:
        n_phi += 1
    D = np.ones((n_phi, n_theta))

    sigma = projected_area(D, isotropic, projected=False)
    error = np.abs(sigma - 2 * np.pi)

    assert((error < 1e-7).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test04_sigma_isotropic(filename):
    # Read raw measurement data from disk
    import pickle
    data = pickle.load(open(filename, "rb"))

    # Create input values
    isotropic = data['isotropic']

    # Get NDF and sigma values
    D = data['ndf_interp']
    sigma = data['sigma']

    sigma_c = projected_area(D, isotropic)

    error = np.abs(sigma - sigma_c)

    assert(sigma.shape == sigma_c.shape)
    assert((error < 1e-4).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test05_ndf_sampler(filename):
    # Read raw measurement data from disk
    import pickle
    data = pickle.load(open(filename, "rb"))

    # Get measurement values
    D_intp = data['ndf_interp']
    D_sampler = data['ndf_sampler']

    D_sampler_c = ndf_intp2sample(D_intp)
    error = np.abs(D_sampler - D_sampler_c)#/div)

    assert(D_sampler.shape == D_sampler_c.shape)
    assert((error < 1e-4).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test06_vndf_sampler(filename):
    # Read raw measurement data from disk
    import pickle
    data = pickle.load(open(filename, "rb"))

    # Get measurement values
    Dvis_intp = data['vndf_interp']
    Dvis_sampler = data['vndf_sampler']

    Dvis_sampler_c = vndf_intp2sample(Dvis_intp)
    error = np.abs(Dvis_sampler - Dvis_sampler_c)

    assert(Dvis_sampler.shape == Dvis_sampler_c.shape)
    assert((error < 1e-4).all())


"""
TODO:   Fix error in NDF/slope calculation.
        NDF/slope calculation not exact yet!
        That's why maximum error is currently still large.
"""
@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test07_ndf_isotropic(filename):
    # Read raw measurement data from disk
    import pickle
    data = pickle.load(open(filename, "rb"))

    isotropic = data['isotropic']

    # Get dimensions of BRDF measurements
    n_theta = data['retro_theta_res']
    n_phi = data['retro_phi_res']

    frC = data['retro_data']

    # Get measurement values
    D = data['ndf_interp']

    if isotropic:
        n_phi = 125
    # Build kernel matrix
    K = build_ndf_kernel(frC, n_theta, n_phi, isotropic)

    # Compute NDF corresponding to first eigenvector
    _, D_c = power_iteration(K, 4)

    # Reshape to NDF format
    if isotropic:
        D_c = np.vstack((D_c, D_c))  # Stack NDF slice for phi=0
    else:
        D_c = np.reshape(D_c, (n_phi, n_theta))

    # Cosine-weight NDF
    D_cw = np.zeros(D_c.shape)
    theta = np.power(np.linspace(0, 1, n_theta), 2) * (np.pi / 2)
    cos_theta = np.cos(theta)
    for i in range(n_theta):
        D_cw[:, i] = D_c[:, i] * cos_theta[i]
    # Normalize weighted NDF or slopes
    D_cw = normalize_2D(D_cw)
    # Reverse weight NDF
    D_c = np.zeros(D_c.shape)
    for i in range(n_theta):
        if cos_theta[i] > np.power(EPSILON, 4):
            D_c[:, i] = D_cw[:, i] / cos_theta[i]

    error = np.abs(D - D_c)
    #print(np.max(error))

    assert(D.shape == D_c.shape)
    assert((error < 1e-2).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test08_slopes_isotropic(filename):
    # Read raw measurement data from disk
    import pickle
    data = pickle.load(open(filename, "rb"))

    isotropic = data['isotropic']

    # Get dimensions of BRDF measurements
    n_theta = data['retro_theta_res']
    n_phi = data['retro_phi_res']

    frC = data['retro_data']

    # Get measurement values
    D = data['ndf_interp']

    if isotropic:
        n_phi = 125
    # Build kernel matrix
    K = build_slope_kernel(frC, n_theta, n_phi, isotropic)

    # Compute NDF corresponding to first eigenvector
    _, P_c = power_iteration(K, 4)

    # Reshape to NDF format
    if isotropic:
        P_c = np.vstack((P_c, P_c))  # Stack NDF slice for phi=0
    else:
        P_c = np.reshape(P_c, (n_phi, n_theta))

    # Normalize
    P_c = normalize_slopes(P_c, isotropic)
    
    # Get slopes from NDF
    P = np.zeros(D.shape)
    theta = np.power(np.linspace(0, 1, n_theta), 2) * (np.pi / 2)
    cos_theta = np.cos(theta)
    for i in range(n_theta):
        P[:, i] = D[:, i] * np.power(cos_theta[i], 4)

    error = np.abs(P - P_c)
    #print(np.max(error))

    assert(P.shape == P_c.shape)
    assert((error < 1e-2).all())

@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test09_vndf_interpolate(filename):
    # Read raw measurement data from disk
    import pickle
    data = pickle.load(open(filename, "rb"))

    isotropic = data['isotropic']

    # Get measurement values
    D = data['ndf_interp']
    sigma = data['sigma']
    theta_i = data['vndf_theta_i']
    phi_i = data['vndf_phi_i']
    Dvis = data['vndf_interp']

    # Calculate VNDF
    Dvis_c = visible_ndf(D, sigma, theta_i, phi_i, isotropic)

    # Normalize
    Dvis = normalize_4D(Dvis, theta_i, phi_i)
    Dvis_c = normalize_4D(Dvis_c, theta_i, phi_i)

    error = np.abs(np.nan_to_num(Dvis) - np.nan_to_num(Dvis_c))
    #print(np.max(error))

    assert(Dvis.shape == Dvis_c.shape)
    assert((error < 1e-4).all())
