import pytest
import pickle
import numpy as np
from visualize import read_tensor, write_tensor
from brdf import (EPSILON, grid_sample, projected_area, sphere_surface_patch,
                  visible_ndf, build_ndf_kernel, build_slope_kernel,
                  power_iteration, vndf_intp2sample, ndf_intp2sample,
                  normalize_slopes, normalize_2D, normalize_4D,
                  incident_elevation, outgoing_direction, weight_measurements,
                  integrate_spectrum)
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
        if cos_theta[i] > np.power(EPSILON, 3):
            D_c[:, i] = D_cw[:, i] / cos_theta[i]

    nef = D[0, 0] / D_cw[0, 0]      # Norm error factor (Mitsuba norm error)
    error = np.abs(D - D_c * nef)
    #print(np.max(error))

    assert(D.shape == D_c.shape)
    assert((error < 1e-2).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test08_slopes_isotropic(filename):
    # Read raw measurement data from disk
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


# TODO: fix sampling
@pytest.mark.parametrize("filename", ["bin/spectralon_spec.bsdf"])
def test10_incident_elevation(filename):
    # Read a tensor file from disk
    tensor = read_tensor(filename)
    theta_i = tensor['theta_i']
    sigma = tensor['sigma']

    # Calculate incident elevations
    theta_i_c = incident_elevation(8, sigma)
    error = np.abs(theta_i - theta_i_c)
    print(theta_i, theta_i_c)
    assert(theta_i_c.shape == theta_i.shape)
    assert((error < 5e-1).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test11_outgoing_samples_isotropic(filename):
    # Read raw measurement data from disk
    data = pickle.load(open(filename, "rb"))

    # Get dimensions of BRDF measurements
    isotropic = data['isotropic']
    R = 32 if isotropic else 64

    Dvis_sampler = data["vndf_sampler"]
    sigma = data["sigma"]

    # Spectral measurements
    n_theta_i = data['theta_i_res']
    n_phi_i = data['phi_i_res']
    phi_i = np.zeros(n_phi_i)
    theta_i = np.zeros(n_theta_i)
    phi_o = np.zeros((n_phi_i, n_theta_i, R, R))
    theta_o = np.zeros((n_phi_i, n_theta_i, R, R))
    for key, value in data.items():
        if key[0] == 'spectra':
            phi_i[key[1]] = np.radians(value[0][0])
            theta_i[key[2]] = np.radians(value[0][1])
            phi_o[key[1], key[2], key[3], key[4]] = np.radians(value[0][2])
            theta_o[key[1], key[2], key[3], key[4]] = np.radians(value[0][3])

    # Compute sample positions
    theta_o_c, phi_o_c, active = outgoing_direction(R, R, Dvis_sampler, phi_i, theta_i, isotropic)

    # Horizontal slice of samples is interpolated
    # (set to 0 for measured data)
    error_phi_o = phi_o[:, 0:-1] - phi_o_c[:, 0:-1]
    error_theta_o = theta_o[:, 0:-1] - theta_o_c[:, 0:-1]
    assert((error_phi_o < 1e-4).all())
    assert((error_theta_o < 1e-4).all())


@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test12_spectral_wavlengths_isotropic(filename):
    # Read raw measurement data from disk
    data = pickle.load(open(filename, "rb"))

    # Wavelengths (usable spectral range)
    wavelengths = np.array(data['wavelengths'])
    wl_min_idx = np.argmin(np.abs(wavelengths - 359.))
    wl_max_idx = np.argmin(np.abs(wavelengths - 1001.))
    wavelengths = wavelengths[wl_min_idx:wl_max_idx+1]

    # Read tensor file from disc
    reference = filename.rsplit('.', 1)[0] + "_spec.bsdf"
    tensor = read_tensor(reference)
    wavelengths_ref = tensor["wavelengths"]
    
    error = np.abs(wavelengths_ref - wavelengths)
    assert(wavelengths_ref.shape == wavelengths.shape)
    assert((error < 1e-4).all())


# TODO: adapt black/white level for spectral measurements
@pytest.mark.parametrize("filename", ["bin/spectralon.pickle"])
def test13_spectral_measurements_isotropic(filename):
    # Read raw measurement data from disk
    data = pickle.load(open(filename, "rb"))

    # Get dimensions of BRDF measurements
    isotropic = data['isotropic']
    jacobian = data['scale_by_jacobian']
    R = 32 if isotropic else 64

    Dvis_sampler = data["vndf_sampler"]
    D = data["ndf_interp"]
    sigma = data["sigma"]

    # Wavelengths (usable spectral range)
    wavelengths = np.array(data['wavelengths'])
    wl_min_idx = np.argmin(np.abs(wavelengths - 359.))
    wl_max_idx = np.argmin(np.abs(wavelengths - 1001.))
    wavelengths = wavelengths[wl_min_idx:wl_max_idx+1]

    # Read tensor file from disc
    reference = filename.rsplit('.', 1)[0] + "_spec.bsdf"
    tensor = read_tensor(reference)
    spec_ref = tensor["spectra"]

    # Spectral measurements
    n_theta_i = data['theta_i_res']
    n_phi_i = data['phi_i_res']
    phi_i = np.zeros(n_phi_i)
    theta_i = np.zeros(n_theta_i)
    spec = np.zeros((n_phi_i, n_theta_i, wavelengths.size, R, R))
    for key, value in data.items():
        if key[0] == 'spectra':
            phi_i[key[1]] = np.radians(value[0][0])
            theta_i[key[2]] = np.radians(value[0][1])
            spec[int(key[1]), int(key[2]), :, int(key[3]), int(key[4])] = \
            value[1][wl_min_idx:wl_max_idx+1]


    # Compute sample positions
    theta_o, phi_o, active = outgoing_direction(R, R, Dvis_sampler, theta_i, phi_i, isotropic)

    # Jacobian weighted measuremnts
    if jacobian:
        # TODO: figure out how to include black/white level
        spec_c = weight_measurements(spec, D, sigma, theta_i, phi_i,
                                     theta_o, phi_o, active)
        # Horizontal slice of samples is interpolated
        # (set to 0 for measured data)
        error = np.abs(spec_ref[:, 0:-1] - spec_c[:, 0:-1])
        print(error)
        print(error.max())


# TODO: fix integration
@pytest.mark.parametrize("filename", ["bin/spectralon_spec.bsdf"])
def test14_luminance_isotropic(filename):
    # Read tensor file from disc
    tensor = read_tensor(filename)
    spec = tensor["spectra"]
    wavelengths = tensor["wavelengths"]
    luminance = tensor["luminance"]

    # Luminace integration
    luminance_c = integrate_spectrum(spec, wavelengths)
    error = np.abs(luminance - luminance_c)

    assert(luminance.shape == luminance_c.shape)
    assert((error < 5e-1).all())


@pytest.mark.parametrize("filename", ["bin/spectralon_spec.bsdf"])
def test15_save_tensor(filename):
    # Read tensor file from disc
    tensor = read_tensor(filename)

    # Copy dictionary and write to disc
    tensor_c = dict()
    for key in tensor:
        tensor_c[key] = tensor[key]
    output = "bin/out.bsdf"
    write_tensor(output, **tensor_c)
    tensor_c = read_tensor(output)

    # Check dictionaries are equal
    assert([key for key in tensor] == [key for key in tensor_c])
    equal = True
    for key in tensor:
        equal &= (tensor_c[key] == tensor[key]).all()
    assert(equal)




