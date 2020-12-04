import enoki as ek
import numpy as np

"""
Build kernel matrix for calculation of micro-facet Normal Distribution
Function

params:
    @frC = cosign weighted BSDF mesurements
    @n_theta = elevation samples of input
    @n_phi = azimuth samples of input / calculation
    @isotropic = material property isotropic
return:
    @K = kernel matrix
"""
def build_kernel(frC, n_theta, n_phi, isotropic):
    N = frC.size
    # Build kernel matrix
    K = np.zeros((N,N))
    weights = np.ones(N)
    if isotropic:
        if n_theta != N:
            raise("Error: Grid dimensions do not match BRDF mesurements!")
        from scipy.integrate import trapz
        # Interpolation grid
        theta_m = u2theta(np.linspace(0, 1, n_theta))
        phi_m = u2phi(np.linspace(0, 1, n_phi))
        dphi = phi_m[1] - phi_m[0]      # constant phi increment

        # Directions of resulting kernel (only dependant on theta)
        # phi = 0 (arbitrary choice)
        omega = np.column_stack((np.sin(theta_m),
                                 np.zeros(n_theta),
                                 np.cos(theta_m)))
        for j in range(N):
            # Integrate over phi_m [-pi, pi]
            sin_theta_m, cos_theta_m = (np.sin(theta_m[j]), np.cos(theta_m[j]))
            sin_phi_m, cos_phi_m = (np.sin(phi_m), np.cos(phi_m))

            omega_m = np.vstack((cos_phi_m * sin_theta_m,
                                 sin_phi_m * sin_theta_m,
                                 cos_theta_m * np.ones(n_phi)))
            tmp = np.matmul(omega, omega_m).clip(0, None) * sin_theta_m
            integral = trapz(tmp, dx=dphi, axis=1)

            # Calculate kernel
            K[:, j] = weights * frC * integral
    else:
        if (n_theta * n_phi) != N:
            raise("Error: Grid dimensions do not match BRDF mesurements!")
        _, _, omega_m = grid_sample(n_theta, n_phi)
        for j in range(N):
            # Calculate kernel
            K[:, j] = weights * frC * np.dot(omega_m, omega_m[j]).clip(0, None)
    return K

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
    Dvis_sample = np.zeros(Dvis_intp.shape)

    # Create uniform samples and warp by G2 mapping
    n_theta = Dvis_intp.shape[3]
    theta_m = u2theta(np.linspace(0, 1, n_theta))

    # Apply Jacobian correction factor to interpolants
    for l in range(n_theta):
        jc = np.sqrt(8 * np.power(np.pi, 3) * theta_m[l]) * np.sin(theta_m[l])
        Dvis_sample[:, :, :, l] = Dvis_intp[:, :, :, l] * jc / (n_theta * n_theta)
    return Dvis_sample


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
    u_m = np.meshgrid(np.linspace(0, 1, n_theta), np.linspace(0, 1, n_phi))
    theta_m = u2theta(u_m[0]).flatten()
    phi_m = u2phi(u_m[1]).flatten()

    # Create quadrature nodes (Spherical -> Cartesian coordinates)
    sin_theta_m, cos_theta_m = (np.sin(theta_m), np.cos(theta_m))
    sin_phi_m, cos_phi_m = (np.sin(phi_m), np.cos(phi_m))

    omega_m = np.column_stack((cos_phi_m * sin_theta_m,
                               sin_phi_m * sin_theta_m,
                               cos_theta_m))
    return theta_m, phi_m, omega_m