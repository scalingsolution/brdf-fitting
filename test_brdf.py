import pytest

import numpy as np
from visualize import read_tensor
from brdf import (grid_sample, projected_area,
                  sphere_surface_patch, visible_ndf, build_kernel,
                  power_iteration, vndf_intp2sample,
                  ndf_intp2sample, normalize_2D, normalize_4D)

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


@pytest.mark.parametrize("filename", ["bin/spectralon_spec.bsdf"])
def test04_sigma_isotropic(filename):
    # Read a tensor file from disk
    tensor = read_tensor(filename)

    # Create input values
    isotropic = True

    # Get NDF and sigma values
    D = tensor['ndf']
    sigma = tensor['sigma']

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
TODO:   Fix error in NDF calculation.
        NDF calculation not exact yet!
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
    D_intp = data['ndf_interp']

    if isotropic:
        n_phi = 125
    # Build kernel matrix
    K = build_kernel(frC, n_theta, n_phi, isotropic)

    # Compute NDF corresponding to first eigenvector
    _, D_intp_c = power_iteration(K, 4)

    # Reshape to NDF format
    if isotropic:
        D_intp_c = np.vstack((D_intp_c, D_intp_c))  # Stack NDF slice for phi=0
    else:
        D_intp_c = np.reshape(D_intp_c, (n_phi, n_theta))

    # Normalize
    D_intp = normalize_2D(D_intp)
    D_intp_c = normalize_2D(D_intp_c)

    error = np.abs(D_intp - D_intp_c)

    assert(D_intp.shape == D_intp_c.shape)
    assert((error < 1e-2).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test08_vndf_interpolate(filename):
    # Read raw measurement data from disk
    import pickle
    data = pickle.load(open(filename, "rb"))

    isotropic = data['isotropic']

    # Get measurement values
    D_intp = data['ndf_interp']
    sigma = data['sigma']
    theta_i = data['vndf_theta_i']
    phi_i = data['vndf_phi_i']
    Dvis_intp = data['vndf_interp']

    # Calculate VNDF
    Dvis_intp_c = visible_ndf(D_intp, sigma, theta_i, phi_i, isotropic)

    # Normalize
    Dvis_intp = normalize_4D(Dvis_intp, theta_i, phi_i)
    Dvis_intp_c = normalize_4D(Dvis_intp_c, theta_i, phi_i)

    error = np.abs(np.nan_to_num(Dvis_intp) - np.nan_to_num(Dvis_intp_c))

    assert(Dvis_intp.shape == Dvis_intp_c.shape)
    assert((error < 1e-4).all())
