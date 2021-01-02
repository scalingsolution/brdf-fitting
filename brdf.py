import enoki as ek
import numpy as np

# Minimal divison coefficient
EPSILON = 1e-4


"""
Build kernel matrix for calculation of micro-facet
Normal Distribution Function (NDF).
params:
    @frC = cosign weighted BSDF mesurements
    @n_theta = number of elevation samples of input
    @n_phi = number of azimuth samples of input (or calculation if isotropic)
    @isotropic = material property isotropic
return:
    @K = NDF kernel matrix
"""
def build_ndf_kernel(frC, n_theta, n_phi, isotropic):
    N = frC.size
    # Build kernel matrix
    K = np.zeros((N,N))
    weights = np.ones(N)
    if isotropic:
        if n_theta != N:
            raise("Error: Grid dimensions do not match BRDF mesurements!")
        from scipy.integrate import trapz
        # Interpolation grid
        theta = u2theta(np.linspace(0, 1, n_theta))
        phi = u2phi(np.linspace(0, 1, n_phi))
        dphi = 2 * np.pi / (n_phi - 1)      # constant increments (= phi[1]-phi[0])

        # Directions of resulting kernel (only dependant on theta)
        # phi = 0 (arbitrary choice)
        omega = np.column_stack((np.sin(theta),
                                 np.zeros(n_theta),
                                 np.cos(theta)))
        for j in range(N):
            # Integrate over phi_m [-pi, pi]
            sin_phi_m, cos_phi_m = (np.sin(phi), np.cos(phi))

            omega_m = np.vstack((cos_phi_m * omega[j, 0],
                                 sin_phi_m * omega[j, 0],
                                 omega[j, 2] * np.ones(n_phi)))
            tmp = np.matmul(omega, omega_m).clip(0, None) * omega[j, 0]
            integral = trapz(tmp, dx=dphi, axis=1)

            # Calculate kernel
            K[:, j] = weights[j] * frC * integral
    else:
        if (n_theta * n_phi) != N:
            raise("Error: Grid dimensions do not match BRDF mesurements!")
        _, _, omega_m = grid_sample(n_theta, n_phi)
        for j in range(N):
            # Calculate kernel
            K[:, j] = weights[j] * frC * np.dot(omega_m, omega_m[j]).clip(0, None)
    return K


"""
Build kernel matrix for calculation of slopes of
micro-facet Normal Distribution Function (NDF).
Corresponds to the Probability Density Function (PDF).
params:
    @frC = cosign weighted BSDF mesurements
    @n_theta = number of elevation samples of input
    @n_phi = number of azimuth samples of input (or calculation if isotropic)
    @isotropic = material property isotropic
return:
    @K = PDF kernel matrix
"""
def build_slope_kernel(frC, n_theta, n_phi, isotropic):
    N = frC.size
    # Build kernel matrix
    K = np.zeros((N,N))
    if isotropic:
        if n_theta != N:
            raise("Error: Grid dimensions do not match BRDF mesurements!")
        from scipy.integrate import trapz
        # Interpolation grid
        theta = u2theta(np.linspace(0, 1, n_theta))
        phi = u2phi(np.linspace(0, 1, n_phi))
        dphi = 2 * np.pi / (n_phi - 1)      # constant increments (= phi[1]-phi[0])
        # Directions of resulting kernel (only dependant on theta)
        # phi = 0 (arbitrary choice)
        omega = np.column_stack((np.sin(theta),
                                 np.zeros(n_theta),
                                 np.cos(theta)))
        for j in range(N):
            # Integrate over phi_m [-pi, pi]
            if omega[j, 2] > EPSILON:
                sin_phi_m, cos_phi_m = (np.sin(phi), np.cos(phi))

                omega_m = np.vstack((cos_phi_m * omega[j, 0],
                                     sin_phi_m * omega[j, 0],
                                     omega[j, 2] * np.ones(n_phi)))
                tmp = np.matmul(omega, omega_m).clip(0, None)
                integral = trapz(tmp, dx=dphi, axis=1)

                # Calculate kernel
                K_ = frC * integral / np.power(omega[j, 2], 4)
                K[:, j] = K_ * omega[j, 0] * np.power(omega[:,2], 4)
    else:
        if (n_theta * n_phi) != N:
            raise("Error: Grid dimensions do not match BRDF mesurements!")
        _, phi_m, omega_m = grid_sample(n_theta, n_phi)
        sin_phi_m = np.sin(phi_m)
        for j in range(N):
            if omega_m[j, 2] > EPSILON:
                # Calculate kernel
                K_ = (frC * np.dot(omega_m, omega_m[j]).clip(0, None) 
                      / np.power(omega[j, 2], 4))
                K[:, j] = K_ * np.power(omega[:,2], 4)           
    return K




"""
Normalize NDF slopes.
Integral over R^2 space of PDF must equal 1.
Equivalent to integral over hemisphere of
NDF * cos(theta).
params:
    @P_in = NDF slopes
    @isotropic = material property isotropic
return:
    @P = normalized NDF slopes
"""
def normalize_slopes(P_in, isotropic):
    from scipy.integrate import trapz
    # Get dimensions from slope PDF
    P = P_in
    n_theta = P.shape[1]
    n_phi = P.shape[0]
    theta = u2theta(np.linspace(0, 1, n_theta))
    phi = u2theta(np.linspace(0, 1, n_theta))
    if isotropic:
        dphi = 2 * np.pi
        tmp = np.zeros(n_theta)
        for i in range(n_theta):
            r = np.tan(theta[i])
            cos_theta = np.cos(theta[i])
            if cos_theta > EPSILON:
                tmp[i] = r * P[0, i] / np.power(cos_theta, 2)
        integral = trapz(tmp, theta)  # integrate over dtheta
        integral *= dphi  # integrate over dphi
    else:
        dphi = 2 * np.pi / (n_phi - 1)  # constant increments (phi[1]-phi[0])
        integral = 0.
        for i in range(n_phi):
            tmp = np.zeros(n_theta)
            for j in range(n_theta):
                r = np.tan(theta[j])
                cos_theta = np.cos(theta[j])
                if cos_theta > EPSILON:
                    tmp[j] = r * P[i, j] / np.power(cos_theta, 2)
            integral += trapz(tmp, theta)  # integrate over dtheta
        # TODO: check factor 2
        integral *= dphi  # integrate over dphi
    # Normalize PDF
    assert(integral > 0.)
    integral = 1. / integral
    P *= integral
    return P

"""
Fit Beckmann parameters to normalized NDF slopes.
Mean surface normals for non-centeral BRDFs are
currently only implemented for anisotropic materials. 
params:
    @P_in = normalized NDF slopes
    @isotropic = material property isotropic
return:
    @a_x = x-dirction RMS slope
    @a_y = y-dirction RMS slope
    @rho = correlation coefficient
    @x_n = mean surface normal x-component
    @y_n = mean surface normal y-component
"""
def beckmann_parameters(P_in, isotropic):
    from scipy.integrate import trapz
    # Get dimensions from slope PDF
    P = P_in
    n_theta = P.shape[1]
    n_phi = P.shape[0]
    theta = u2theta(np.linspace(0, 1, n_theta))
    phi = u2theta(np.linspace(0, 1, n_theta))
    if isotropic:
        tmp = np.zeros(n_theta)
        for i in range(n_theta):
            r = np.tan(theta[i])
            cos_theta = np.cos(theta[i])
            if cos_theta > EPSILON:
                tmp[i] = np.power(r, 3) * P[0, i] / np.power(cos_theta, 2)
        integral = trapz(tmp, theta)
        integral *= np.pi               # = int_0^2pi(cos^2(phi))dphi
        a_x = np.sqrt(2. * integral)
        a_y = a_x
        rho = 0
        # TODO: add shear parameter to isotropic
        x_n = 0
        y_n = 0
    else:
        E = np.zeros(5)                 # moments for extracting Beckmann params
        dphi = 2 * np.pi / (n_phi - 1)  # constant increments (phi[1]-phi[0])
        for i in range(n_phi):
            cos_phi = np.cos(phi[i])
            sin_phi = np.sin(phi[i])
            tmp = np.zeros((5, n_theta))
            for j in range(n_theta):
                r = np.tan(theta[j])
                r_sqr = np.power(r, 2)
                cos_theta = np.cos(theta[j])
                sin_theta = np.sin(theta[j])
                if cos_theta > EPSILON:
                    tmp1 = r * P[i, j] / np.power(cos_theta, 2)
                    tmp[0, j] = tmp1 * -r * cos_phi                     # x_n
                    tmp[1, j] = tmp1 * -r * sin_phi                     # y_n
                    tmp[2, j] = tmp1 * r_sqr * np.power(cos_phi, 2)
                    tmp[3, j] = tmp1 * r_sqr * np.power(sin_phi, 2) 
                    tmp[4, j] = tmp1 * r_sqr * sin_phi * cos_phi 
            E += trapz(tmp, theta, axis=1)  # integrate over dtheta
        # TODO: check factor 2
        E *= dphi   # integrate over dphi
        x_n = E[0]
        y_n = E[1]
        a_x = np.sqrt(2. * (E[2] - np.power(x_n, 2)))
        a_y = np.sqrt(2. * (E[3] - np.power(y_n, 2)))
        rho = 2. * (E[4] - x_n * y_n) / (a_x * a_y)
    return [a_x, a_y, rho, x_n, y_n]


"""
Fit GGX parameters to normalized NDF slopes.
Mean surface normals for non-centeral BRDFs are
currently only implemented for anisotropic materials. 
params:
    @P_in = normalized NDF slopes
    @isotropic = material property isotropic
return:
    @a_x = x-dirction RMS slope
    @a_y = y-dirction RMS slope
    @rho = correlation coefficient (not implemented)
    @x_n = mean surface normal x-component
    @y_n = mean surface normal y-component
"""
def ggx_parameters(P_in, isotropic):
    from scipy.integrate import trapz
    # Get dimensions from slope PDF
    P = P_in
    n_theta = P.shape[1]
    n_phi = P.shape[0]
    theta = u2theta(np.linspace(0, 1, n_theta))
    phi = u2theta(np.linspace(0, 1, n_theta))
    if isotropic:
        dphi = 2 * np.pi
        tmp = np.zeros(n_theta)
        for i in range(n_theta):
            r = np.tan(theta[i])
            cos_theta = np.cos(theta[i])
            if cos_theta > EPSILON:
                tmp[i] = np.power(r, 2) * P[0, i] / np.power(cos_theta, 2)
        integral = trapz(tmp, theta)
        integral *= 4                   # = int_0^2pi(|cos(phi)|)dphi
        a_x = integral
        a_y = a_x
        rho = 0
        # TODO: add shear parameter to isotropic
        x_n = 0
        y_n = 0
    else:
        E = np.zeros(5)                 # moments for extracting Beckmann params
        dphi = 2 * np.pi / (n_phi - 1)  # constant increments (phi[1]-phi[0])
        for i in range(n_phi):
            cos_phi = np.cos(phi[i])
            sin_phi = np.sin(phi[i])
            tmp = np.zeros((5, n_theta))
            for j in range(n_theta):
                r = np.tan(theta[j])
                r_sqr = np.power(r, 2)
                cos_theta = np.cos(theta[j])
                sin_theta = np.sin(theta[j])
                if cos_theta > EPSILON:
                    tmp1 = r * P[i, j] / np.power(cos_theta, 2)
                    tmp[0, j] = tmp1 * -r * cos_phi         # x_n
                    tmp[1, j] = tmp1 * -r * sin_phi         # y_n
                    tmp[2, j] = abs(tmp[0, j])
                    tmp[3, j] = abs(tmp[1, j]) 
                    tmp[4, j] = 0                           # TODO
            E += trapz(tmp, theta, axis=1)  # integrate over dtheta
        # TODO: check factor 2
        E *= dphi   # integrate over dphi
        x_n = E[0]
        y_n = E[1]
        a_x = np.sqrt(np.power(E[2], 2) - np.power(x_n, 2))
        a_y = np.sqrt(np.power(E[3], 2) - np.power(y_n, 2))
        rho = 0     # TODO
    return [a_x, a_y, rho, x_n, y_n]


"""
Evaluate micro-facet distribution model.
Samples are warped by G2 mapping before
converting to Carthesian direction and
evaluating distribution model.
params:
    @n_theta = sample elevation resolution
    @n_phi = sample azimuth resolution (if isotropic: ignored)
    @alpha = roughness parameter (if isotropic: [alpha_x, alpha_y])
    @isotropic = material property isotropic
    @md_type = micro-facet model type ('beckmann' or 'ggx')
return:
    @D = sampled micro-facet distribution
"""
def eval_md_model(n_theta, n_phi, alpha, isotropic, md_type="beckmann"):
    from mitsuba.render import MicrofacetDistribution, MicrofacetType
    if md_type == "beckmann":
        md_t = MicrofacetType.Beckmann
    elif md_type == "ggx":
        md_t = MicrofacetType.GGX
    else:
        print("WARNING: Unknown micro-facet type, returning None.")
        return None
    if isotropic:
        m_D = MicrofacetDistribution(md_t, alpha, False)
        _, _, omega = grid_sample(n_theta, 1)
        D = m_D.eval(omega)
        D = np.vstack((D, D))
    else:
        m_D = MicrofacetDistribution(md_t, alpha[0], alpha[1], False)
        _, _, omega = grid_sample(n_theta, n_phi)
        D = m_D.eval(omega)
        D = np.reshape(D, (n_phi, n_theta))
    #print(alpha)
    #print(omega)
    return D


"""
Evaluate micro-facet distribution.
Specified resolution will be warped by G2 mapping,
before converting to Carthesian direction.
params:
    @n_theta = sample elvation resolution
    @n_phi = sample azimuth resolution (if isotropic: ignored)
    @D_in = micro-facet NDF
    @isotropic = material property isotropic
return:
    @D = sampled micro-facet NDF
"""
def eval_md(n_theta, n_phi, D_in, isotropic):
    from mitsuba.render import MarginalContinuous2D0
    m_D = MarginalContinuous2D0(D_in, normalize=False)
    u = np.meshgrid(np.linspace(0, 1, n_theta), np.linspace(0, 1, n_phi))
    u_0 = u[0].flatten()
    u_1 = u[1].flatten()
    sample = Vector2f(u_0, u_1)
    D = m_D.eval(samples)
    if isotropic:
        D = np.vstack((D, D))
    else:
        D = np.reshape(D, (n_phi, n_theta))
    return D


"""
Return major axis of laser spot on probe from
laser beam diameter and incident elevation.
params:
    @d = laser beam diameter
    @theta = incident elevation (on probe)
return:
    laser spot major axis (on probe)
"""
def effective_spot_size(d, theta):
    return d / np.sin(theta) 

"""
Return maximal elevation for retroreflective measurements,
based on laser beam diameter and probe diameter.
params:
    @d = laser beam diameter
    @w = width of probe in rotation direction
return:
    maximal elevation angle (on probe)
"""
def max_elevation(d, w):
    return np.arcsin(d / w)


"""
Compute projected area of micro-facets as nomalisation

params:
    @D = micro-facet Normal Distribution Function
    @isotropic = material property isotropic
    @projected = apply directional foreshortening
return:
    @sigma = projected area of micro-facets
"""
def projected_area(D, isotropic, projected=True):
    from mitsuba.core import Vector2f, Vector3f
    from mitsuba.core import MarginalContinuous2D0

    # Check dimensions of micro-facet model
    sigma = np.zeros(D.shape)

    # Construct projected surface area interpolant data structure
    m_D = MarginalContinuous2D0(D, normalize=False)

    # Create uniform samples and warp by G2 mapping
    if isotropic:
        n_theta = n_phi = D.shape[1]
    else:
        n_phi = D.shape[0]
        n_theta = D.shape[1]
    theta = u2theta(np.linspace(0, 1, n_theta))
    phi = u2phi(np.linspace(0, 1, n_phi))

    # Temporary values for surface area calculation
    theta_mean = np.zeros(n_theta + 1)
    for i in range(n_theta - 1):
        theta_mean[i+1] = (theta[i+1] - theta[i]) / 2. + theta[i]
    theta_mean[-1] = theta[-1]
    theta_mean[0] = theta[0]
   
    """
    Surface area portion of unit sphere.
    Conditioning better for non vectorized approach.
    a  = sphere_surface_patch(1, theta_next, Vector2f(phi[0], phi[1]))
    """
    a    = np.zeros(n_theta)
    for i in range(n_theta):
        a[i]  = sphere_surface_patch(1, theta_mean[i:i+2], phi[-3:-1])


    # Calculate constants for integration
    for j in range(n_phi):
        # Interpolation points
        o = spherical2cartesian(theta, phi[j])
        # Postion for NDF samples
        u0 = theta2u(theta)
        u1 = np.ones(n_theta) * phi2u(phi[j])
        if j == 0:
            omega = o
            u_0 = u0
            u_1 = u1
            area = a / 2
        else:
            omega = np.concatenate((omega, o))
            u_0 = np.concatenate((u_0, u0))
            u_1 = np.concatenate((u_1, u1))
            if j == n_phi-1:
                area = np.concatenate((area, a/2))
            else:
                area = np.concatenate((area, a))
    sample = Vector2f(u_0, u_1)
    D_s = m_D.eval(sample)
    omega = Vector3f(omega)

    P = 1.
    # Calculate projected area of micro-facets
    for i in range(sigma.shape[0]-1):
        for j in range(sigma.shape[1]):
            # Get projection factor from incident and outgoind direction
            if projected:
                # Incident direction
                omega_i = spherical2cartesian(theta[j], phi[i])
                P = ek.max(0, ek.dot(omega, omega_i))

            # Integrate over sphere
            F = P * D_s
            sigma[i, j] = np.dot(F, area)

        if projected:
            # Normalise
            sigma[i] = sigma[i] / sigma[i, 0]

    # TODO: Check for anisotropic case
    if isotropic:
        sigma[1] = sigma[0]

    return sigma


def spherical2cartesian(theta, phi):
    # Convert: Spherical -> Cartesian coordinates
    [sin_theta, cos_theta]  = ek.sincos(theta)
    [sin_phi, cos_phi]      = ek.sincos(phi)

    from mitsuba.core import Vector3f
    omega = Vector3f(cos_phi * sin_theta,
                     sin_phi * sin_theta,
                     cos_theta)
    return omega


def elevation(d):
    dist = ek.sqrt(ek.sqr(d.x) + ek.sqr(d.y) + ek.sqr(d.z - 1.))
    return 2. * ek.safe_asin(.5 * dist);


def cartesian2spherical(w):
    # Convert: Cartesian coordinates -> Spherical
    theta  = elevation(w) #ek.acos(w.z)
    phi  = ek.atan2(w.y, w.x)
    phi = ek.select(phi+ek.pi < 1e-4, ek.pi, phi)
    return theta, phi


def sphere_surface_patch(r, dtheta, dphi):
    # Hemisphere surface area
    h = 2 * np.pi * np.square(r)
    # Elevation slice
    el_s = np.cos(dtheta[0]) - np.cos(dtheta[1])
    # Azimimuth slice
    az_s = (dphi[1] - dphi[0]) / (2 * np.pi)
    return h * el_s * az_s


"""
Compute visible (bidirectional) micro-facet Normal Distribution Function (NDF)

params:
    @D = micro-facet Normal Distribution Function
    @sigma = projected area of micro-facets
    @theta_i = incident elevation
    @phi_i = incident azimuth
    @isotropic = material property isotropic
return:
    @Dvis visible NDF
"""
def visible_ndf(D, sigma, theta_i, phi_i, isotropic):
    from mitsuba.core import Vector2f
    from mitsuba.core import MarginalContinuous2D0

    # Construct projected surface area interpolant data structure
    m_sigma = MarginalContinuous2D0(sigma, normalize=False)

    # Create uniform samples and warp by G2 mapping
    if isotropic:
        n_theta = n_phi = D.shape[1]
    else:
        n_phi = D.shape[0]
        n_theta = D.shape[1]

    # Check dimensions of micro-facet model
    Dvis = np.zeros((phi_i.size, theta_i.size, n_phi, n_theta))

    theta = u2theta(np.linspace(0, 1, n_theta))
    phi = u2phi(np.linspace(0, 1, n_phi))

    # Calculate projected area of micro-facets
    for i in range(Dvis.shape[0]):          # incident elevation
        for j in range(Dvis.shape[1]):      # incident azimuth
            # Postion for sigma samples
            sample = Vector2f(theta2u(theta_i[j]), phi2u(phi_i[i]))
            sigma_i = m_sigma.eval(sample)

            # Incident direction
            omega_i = spherical2cartesian(theta_i[j], phi_i[i])
            #print(np.degrees(theta_i[j]), np.degrees(phi_i[i]))

            for k in range(Dvis.shape[2]):  # observation azimuth
                # Observation direction
                omega = spherical2cartesian(theta, phi[k])
                sample = Vector2f(theta2u(theta), phi2u(phi[k]))

                # NDF at observation directions
                if isotropic:
                    D_m = D[0]
                else:
                    D_m = D[k]
                # Bidirectional NDF
                Dvis[i, j, k] = ek.max(0, ek.dot(omega, omega_i)) * D_m / sigma_i
    return Dvis


def vndf_intp2sample(Dvis_intp):
    # Check dimensions of micro-facet model
    Dvis_sampler = np.zeros(Dvis_intp.shape)

    # Create uniform samples and warp by G2 mapping
    n_theta = Dvis_intp.shape[3]
    theta_m = u2theta(np.linspace(0, 1, n_theta))

    # Apply Jacobian correction factor to interpolants
    for l in range(n_theta):
        jc = np.sqrt(8 * np.power(np.pi, 3) * theta_m[l]) * np.sin(theta_m[l])
        Dvis_sampler[:, :, :, l] = Dvis_intp[:, :, :, l] * jc / (n_theta * n_theta)
    return Dvis_sampler


def ndf_intp2sample(D_intp):
    # Check dimensions of micro-facet model
    D_sampler = np.zeros(D_intp.shape)

    # Create uniform samples and warp by G2 mapping
    n_theta = D_intp.shape[1]
    theta_m = u2theta(np.linspace(0, 1, n_theta))

    # Apply Jacobian correction factor to interpolants
    for l in range(n_theta):
        jc = np.sqrt(8 * np.power(np.pi, 3) * theta_m[l]) * np.sin(theta_m[l])
        D_sampler[:, l] = D_intp[:, l] * jc / n_theta
    return D_sampler


def normalize_4D(F, theta_i, phi_i):
    # Normalize function so that integral = 1
    from mitsuba.core import Vector2f
    from mitsuba.core import MarginalContinuous2D2
    F_norm = np.zeros(F.shape)
    params = [phi_i.tolist(), theta_i.tolist()]
    # Construct projected surface area interpolant data structure
    m_F_norm = MarginalContinuous2D2(F, params, normalize=True)

    # Create uniform samples
    u_1 = np.linspace(0, 1, F_norm.shape[3])
    u_2 = np.linspace(0, 1, F_norm.shape[2])

    # Sample normalized mapping
    for i in range(F_norm.shape[0]):
        for j in range(F_norm.shape[1]):
            for k in range(F_norm.shape[2]):
                sample = Vector2f(u_1, u_2[k])
                F_norm[i, j, k] = m_F_norm.eval(sample, [phi_i[i], theta_i[j]])
    return F_norm


def normalize_2D(F):
    # Normalize function so that integral = 1
    from mitsuba.core import Vector2f
    from mitsuba.core import MarginalContinuous2D0
    F_norm = np.zeros(F.shape)
    # Construct projected surface area interpolant data structure
    m_F_norm = MarginalContinuous2D0(F, normalize=True)

    # Create uniform samples
    u_1 = np.linspace(0, 1, F_norm.shape[1])
    u_2 = np.linspace(0, 1, F_norm.shape[0])

    # Sample normalized mapping
    for k in range(F_norm.shape[0]):
        sample = Vector2f(u_1, u_2[k])
        F_norm[k] = m_F_norm.eval(sample)
    return F_norm


def brdf_samples(vndf, theta_i, phi_i, n_theta, n_phi):
    from mitsuba.core import MarginalContinuous2D2
    # Construct projected surface area interpolant data structure
    params = [phi_i.tolist(), theta_i.tolist()]
    m_vndf = MarginalContinuous2D2(vndf, params, normalize=False)

    u_m = np.meshgrid(np.linspace(0, 1, n_theta), np.linspace(0, 1, n_phi))
    u_0 = u_m[0].flatten()
    u_1 = u_m[1].flatten()
    samples = Vector2f(u_0, u_1)

    # Check dimensions of micro-facet model
    theta_o = phi_o = np.zeros((phi_i.size, theta_i.size, n_phi * n_theta))

    # Warp sample grid to VNDF
    for i in range(vndf.shape[0]):
        for j in range(vndf.shape[1]):
            val = m_vndf.eval(samples, [theta_i[j], phi_i[i]])
            m = m_vndf.sample(samples, [theta_i[j], phi_i[i]])
            # Map samples to spere
            theta_o[i, j] = u2theta(m[0])
            phi_o[i, j] = u2phi(m[1])
    return theta_o, phi_o


def u2theta(u):
    return np.square(u) * (np.pi / 2.)


def u2phi(u):
    return (2. * u - 1.) * np.pi


def theta2u(theta):
    return np.sqrt(theta * (2. / np.pi))


def phi2u(phi):
    return (0.5 * (phi / np.pi + 1))


def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


def power_iteration(A, M, eigen=False):
    n, d = A.shape
    ev = None
    v = np.ones(d)
    ev = eigenvalue(A, v)

    for i in range(M):
        v = np.matmul(A, v)
    if eigen:
        ev = eigenvalue(A, v)
    return ev, v


"""
Generate uniform samples on grid U^2 and map them to sphere S^2.
Apply trasformation G2.

params:
    @n_theta = uniform samples on u1 (theta dimension)
    @n_phi = uniform samples on u2 (phi dimension)
return
    @theta_m = elevation samples
    @phi_m = azimuth samples
    @omega_m = ray directions
"""
def grid_sample(n_theta, n_phi):
    # Create uniform samples and warp by G2 mapping
    u = np.meshgrid(np.linspace(0, 1, n_theta), np.linspace(0, 1, n_phi))
    theta = u2theta(u[0]).flatten()
    phi = u2phi(u[1]).flatten()

    # Create quadrature nodes (Spherical -> Cartesian coordinates)
    sin_theta, cos_theta = (np.sin(theta), np.cos(theta))
    sin_phi, cos_phi = (np.sin(phi), np.cos(phi))

    omega = np.column_stack((cos_phi * sin_theta,
                             sin_phi * sin_theta,
                             cos_theta))
    return theta, phi, omega


"""
Generate incident elevation samples based on inverse sigma distribution.

params:
    @n = number of samples for incident direction
    @sigma = projected area of micro-facets
return
    @theta_i = incident elevations
"""
def incident_elevation(n, sigma):
    from mitsuba.core import Float, Vector2f
    from mitsuba.core import MarginalContinuous2D0
    # Construct projected surface area interpolant data structure
    sigma_sampler = ndf_intp2sample(sigma)
    m_sigma = MarginalContinuous2D0(sigma_sampler, normalize=False)

    # Warp samples by projected area
    samples = Vector2f(np.linspace(0, 1, n), 0.5)
    u_m, _ = m_sigma.sample(samples)

    # Map samples to sphere
    theta_i = u2theta(u_m[0]) #(np.pi - u_m[0] * np.pi) / 2
    return theta_i


"""
Compute outgoing direction samples from visible NDF.

params:
    @n_phi = number of outgoing azimuth samples per incident direction slice
    @n_phi = number of outgoing elevation samples per incident direction slice
    @Dvis_sampler = jacobian corrected visible NDF for sample generation
    @phi_i = incident azimuth [rad]
    @theta_i = incident elevation [rad]
    @isotropic = material property isotropic
return
    @theta_o = outgoing elevation [rad]
    @phi_o = outgoing azimuth [rad]
    @active = valid outgoing directions
"""
def outgoing_direction(n_phi, n_theta, Dvis_sampler, phi_i, theta_i, isotropic):
    from mitsuba.core import Vector2f, Frame3f
    from mitsuba.core import MarginalContinuous2D2

    MAX_ELEVATION = 85
    phi_o = np.zeros((phi_i.size, theta_i.size, n_phi, n_theta))
    theta_o = np.zeros((phi_i.size, theta_i.size, n_phi, n_theta))
    active = np.ones((phi_i.size, theta_i.size, n_phi, n_theta), dtype='bool')

    # Create uniform samples
    u_0 = np.linspace(0, 1, n_theta)
    u_1 = np.linspace(0, 1, n_phi)
    samples = Vector2f(np.tile(u_0, n_phi), np.repeat(u_1, n_theta))

    # Construct projected surface area interpolant data structure
    params = [phi_i.tolist(), theta_i.tolist()]
    m_vndf = MarginalContinuous2D2(Dvis_sampler, params, normalize=True)

    for i in range(phi_i.size):
        for j in range(theta_i.size):
            # Warp uniform samples by VNDF distribution (G1 mapping) 
            u_m, ndf_pdf = m_vndf.sample(samples, [phi_i[i], theta_i[j]])
            # Convert samples to radians (G2 mapping)
            theta_m = u2theta(u_m.x)    # [0, 1] -> [0, pi]
            phi_m = u2phi(u_m.y)        # [0, 1] -> [0, 2pi]
            if isotropic:
                phi_m += phi_i[i]               
            # Phase vector
            m = spherical2cartesian(theta_m, phi_m)
            # Incident direction
            wi = spherical2cartesian(theta_i[j], phi_i[i])
            # Outgoing direction (reflection over phase vector)
            wo = ek.fmsub(m, 2.0 * ek.dot(m, wi), wi)
            tmp1, tmp2 = cartesian2spherical(wo)
            # Remove invalid directions
            act = Frame3f.cos_theta(wo) > 0
            act &= tmp1 < (MAX_ELEVATION / 180 * np.pi)
            if isotropic:
                act &= tmp2 > 0 #, np.abs(tmp2) np.logical_or( - np.pi < 1e-4)
            # Fit to datashape
            act = np.reshape(act, (n_phi, n_theta))
            tmp1 = np.reshape(tmp1, (n_phi, n_theta))
            tmp2 = np.reshape(tmp2, (n_phi, n_theta))
            tmp1[~act] = 0
            tmp2[~act] = 0
            theta_o[i, j] = tmp1
            phi_o[i, j] = tmp2
            active[i, j] = act
    return [theta_o, phi_o, active]


"""
Weight spectral measurements by inverse jacobian of mapping.

params:
    @spec = spectral measurements
    @D = micro-facet Normal Distribution Function (NDF)
    @sigma = projected area of micro-facets
    @phi_i = incident azimuth [rad]
    @theta_i = incident elevation [rad]
    @theta_o = outgoing elevation [rad]
    @phi_o = outgoing azimuth [rad]
    @active = valid outgoing directions
return
    @scaled = weighted spectral measurements
"""
def weight_measurements(spec, D, sigma, phi_i, theta_i, phi_o, theta_o, active):
    from mitsuba.core import MarginalContinuous2D0, Vector2f
    m_ndf = MarginalContinuous2D0(D, normalize=False)
    m_sigma = MarginalContinuous2D0(sigma, normalize=False)
    scaled = np.zeros(spec.shape)
    n_wavelengths = spec.shape[2]
    n_phi = spec.shape[3]
    n_theta = spec.shape[4]

    for i in range(phi_i.size):
        for j in range(theta_i.size):
            # Incient direction
            wi = spherical2cartesian(theta_i[j], phi_i[i])
            u_wi = Vector2f(theta2u(theta_i[j]), phi2u(phi_i[i]))   
            # Outgoing direction
            wo = spherical2cartesian(theta_o[i, j].flatten(), phi_o[i, j].flatten())
            # Phase direction
            m = ek.normalize(wi + wo)
            theta_m, phi_m = cartesian2spherical(wo)
            u_m = Vector2f(theta2u(theta_m), phi2u(phi_m))
            # Scale by inverse jacobian
            jacobian = m_ndf.eval(u_m) / (4 * m_sigma.eval(u_wi))
            jacobian = np.reshape(jacobian, (n_phi, n_theta))
            for k in range(n_wavelengths): 
                scaled[i, j, k] = spec[i, j, k] / jacobian
    for k in range(n_wavelengths): 
        scaled[:, :, k][~active] = 0 
    return scaled


"""
Calculate luminoscity of spectral measurements, by integrating
over wavelength range.

params:
    @spec = spectral measurements
    @wavelengths = wavelengths used for spectral measurements
return
    @luminoscity = luminoscity
"""
def integrate_spectrum(spec, wavelengths):
    from scipy.integrate import trapz
    n_phi_i = spec.shape[0]
    n_theta_i = spec.shape[1]
    n_phi_o = spec.shape[3]
    n_theta_o = spec.shape[4]
    luminoscity = np.zeros((n_phi_i, n_theta_i, n_phi_o, n_theta_o))
    x = wavelengths #np.repeat(wavelengths, n_phi_o * n_theta_o)
    span = x.max() - x.min()
    for i in range(n_phi_i):
        for j in range(n_theta_i):
            y = spec[i, j]
            y.reshape(-1, *y.shape[-2:])
            integral = trapz(y, x, axis=0) / span
            luminoscity[i, j] = np.reshape(integral, (n_phi_o, n_theta_o))
    return luminoscity