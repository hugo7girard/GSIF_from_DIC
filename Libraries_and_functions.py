#%% Libraries

import pyvista as pv
import numpy as np
import vtk
from pyvista.utilities.helpers import wrap
import matplotlib.pyplot as plt
from PIL import Image

#%% Functions

def generate_circle(
    x_center: float,
    y_center: float,
    radius: float,
    n_points: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
    
    """
    Generate a circular contour in the XY plane.

    Parameters
    ----------
    x_center : float
        X-coordinate of the circle center (in pixels).
    y_center : float
        Y-coordinate of the circle center (in pixels).
    radius : float
        Radius of the circle (in pixels).
    n_points : int
        Number of points along the circumference.

    Returns
    -------
    circle_points : np.ndarray
        Array of shape (n_points, 3) with XYZ coordinates of the circle points.
    theta : np.ndarray
        Array of angles corresponding to each point (in radians).
    dL : float
        Arc length between two consecutive points (in pixels).
    """
    
    # center location in in pixel
    center = [x_center,y_center, 0.0] 
    
    # Angles from 0 to just below 2π
    theta = np.linspace(0, 2*np.pi - 2*np.pi/(n_points), n_points)
    
    # Arc length between two points
    dL = 2*np.pi/(n_points)*radius
    
    # Circle points in 3D (constant Z)
    circle_points = np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        np.full_like(theta, center[2])  # Z = 0
    ])
    return circle_points,theta,dL

def compute_U_E_DIC(
    file_path: str,
    circle_points: np.ndarray,
    scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pv.PolyData]:
    
    """
    Interpolate displacement and strain data from a mesh onto given points.

    This function uses a VTK ProbeFilter to sample displacement and strain
    values from a PyVista mesh at the provided `circle_points`. Displacements
    are scaled to physical units (e.g., mm) using `scale`.

    Parameters
    ----------
    file_path : str
        Path to the mesh .vtk file containing displacement ('U') and strain ('Strain') data.
    circle_points : np.ndarray
        Array of shape (n_points, 3) with XYZ coordinates where interpolation is performed.
    scale : float
        Scaling factor to convert displacement units (e.g., from pixels to mm).

    Returns
    -------
    Ux_scaled : np.ndarray
        X-component of displacement at each circle point (scaled).
    Uy_scaled : np.ndarray
        Y-component of displacement at each circle point (scaled).
    Exx : np.ndarray
        Strain in the X-direction at each circle point.
    Exy : np.ndarray
        Shear strain at each circle point.
    Eyy : np.ndarray
        Strain in the Y-direction at each circle point.
    circle : pv.PolyData
        PyVista PolyData object representing the input points.
    """
    
    # Load mesh
    mesh = pv.read(file_path)
    
    # Create PolyData from given circle points
    circle = pv.PolyData(circle_points)
    
    # Use VTK ProbeFilter to interpolate mesh data at circle points
    probe = vtk.vtkProbeFilter()
    probe.SetSourceData(mesh)
    probe.SetInputData(circle)
    probe.Update()
    
    # Convert VTK output to PyVista
    sampled = wrap(probe.GetOutput())
    
    # Extract displacement and strain components
    Ux = sampled["U"][:, 0]  
    Uy = sampled["U"][:, 1]
    Exx = sampled["Strain"][:, 0]
    Exy = sampled["Strain"][:, 1]
    Eyy = sampled["Strain"][:, 4]
    
    # Convert displacement to physical units
    Ux_scaled = Ux * scale
    Uy_scaled = Uy * scale
    
    return Ux_scaled,Uy_scaled,Exx,Exy,Eyy,circle

def compute_S_DIC(
    Exx: np.ndarray,
    Eyy: np.ndarray,
    Exy: np.ndarray,
    E: float,
    nu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Compute stress components from strain data assuming linear elastic isotropy.

    Uses Hooke's law for an isotropic material in 2D, based on the given Young's
    modulus `E` and Poisson's ratio `nu`.

    Parameters
    ----------
    Exx : np.ndarray
        Strain in the X-direction at each circle point.
    Eyy : np.ndarray
        Strain in the Y-direction at each circle point.
    Exy : np.ndarray
        Shear strain at each circle point.
    E : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.

    Returns
    -------
    Sxx : np.ndarray
        Normal stress in the X-direction.
    Syy : np.ndarray
        Normal stress in the Y-direction.
    Sxy : np.ndarray
        Shear stress.
    """
    
    Sxx = E/(1+nu) *( Exx + nu/(1-2*nu)*(Exx + Eyy))
    Syy = E/(1+nu) *( Eyy + nu/(1-2*nu)*(Exx + Eyy))
    Sxy = E/(1+nu) *( Exy ) 
    
    return Sxx,Syy,Sxy

def show_image_ref_and_fields(
    file_path: str,
    ref_image_path: str,
    output: str,
    center_x: float,
    center_y: float,
    radius: float,
    nb_points: int
    ) -> None:
    
    """
    Display a reference image overlaid with a mesh field and a circular contour.

    This function:
    1. Loads a mesh file containing displacement and strain fields.
    2. Loads and flips the reference image vertically to match mesh coordinates.
    3. Overlays the mesh field (`output`) on top of the reference image.
    4. Draws a circular contour for visualization.

    Parameters
    ----------
    file_path : str
        Path to the .vtk file containing displacement ('Displacement', 'U') and strain ('Strain') arrays.
    ref_image_path : str
        Path to the reference image file (e.g., .png, .jpg).
    output : str
        Name of the mesh array to display as a scalar field (e.g., "Ux", "Ux_no_rbm", "Exx").
    center_x : float
        X-coordinate of the circular contour center (in pixels).
    center_y : float
        Y-coordinate of the circular contour center (in pixels).
    radius : float
        Radius of the circular contour (in pixels).
    nb_points : int
        Number of points along the circular contour.

    Returns
    -------
    None
        Displays the PyVista interactive window.
    """
    
    # Load mesh and extract relevant fields
    mesh = pv.read(file_path)
    mesh["Ux"] = mesh["Displacement"][:, 0]
    mesh["Uy"] = mesh["Displacement"][:, 1]
    mesh["Ux_no_rbm"] = mesh["U"][:, 0]
    mesh["Uy_no_rbm"] = mesh["U"][:, 1]
    mesh["Exx"] = mesh["Strain"][:, 0]
    mesh["Exy"] = mesh["Strain"][:, 1]
    mesh["Eyy"] = mesh["Strain"][:, 4]
    
    # Load and process reference image
    image = Image.open(ref_image_path).convert("RGB")  # PIL image
    img_array = np.array(image)
    img_array_flipped = np.flipud(img_array) #flip of the y-axis
    ny, nx, _ = img_array_flipped.shape
    #extent = [0, nx, 0, ny]
    
    # Create background image as a PyVista object
    background = pv.ImageData(dimensions=(nx, ny, 1))
    background.spacing = (1, 1, 1)
    background.origin = (0, 0, 0)
    background.point_data["RGB"] = img_array_flipped.reshape((-1, 3))
    
    # Generate circular contour
    circle_points,theta,dL = generate_circle(center_x,center_y,radius,nb_points)
    circle = pv.PolyData(circle_points)
    
    # Create PyVista plot
    p = pv.Plotter()
    p.add_mesh(background, rgb=True)
    p.add_mesh(mesh, scalars=output, cmap="jet", opacity=0.6, show_scalar_bar=True)
    p.add_mesh(circle, color='red', line_width=3, label="Circular Contour")
    p.show_grid()
    p.show()


def remove_point_out_rota(
    Ux: np.ndarray,
    Uy: np.ndarray,
    Exx: np.ndarray,
    Eyy: np.ndarray,
    Exy: np.ndarray,
    theta: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
    rota: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Remove invalid points (where there is no field so Ux == 0) and rotate displacement/strain fields.

    This function:
    1. Filters out points where the X-displacement (`Ux`) is exactly zero.
    2. Rotates displacement vectors and strain tensors by an angle `rota`.
    3. Adjusts `theta` values by subtracting the rotation angle and rearranging
       them into the range [-π, π].

    Parameters
    ----------
    Ux, Uy : np.ndarray
        Displacement components in the original coordinate system.
    Exx, Eyy, Exy : np.ndarray
        Strain tensor components in the original coordinate system.
    theta : np.ndarray
        Angular position of each point (in radians).
    x_points, y_points : np.ndarray
        Coordinates of each point.
    rota : float
        Rotation angle (in radians), positive counterclockwise.

    Returns
    -------
    Ux_rot, Uy_rot : np.ndarray
        Displacement components in the rotated coordinate system.
    Exx_rot, Eyy_rot, Exy_rot : np.ndarray
        Strain tensor components in the rotated coordinate system.
    theta_rearranged : np.ndarray
        Adjusted angular positions in the range [-π, π].
    x_clean, y_clean : np.ndarray
        Filtered coordinates of valid points.
    """
    
    # Create mask to filter out invalid points
    mask = [u != 0.0 for u in Ux]
    Ux_clean = np.array([u for u, m in zip(Ux, mask) if m])
    Uy_clean = np.array([v for v, m in zip(Uy, mask) if m])
    Exx_clean = np.array([v for v, m in zip(Exx, mask) if m])
    Eyy_clean = np.array([v for v, m in zip(Eyy, mask) if m])
    Exy_clean = np.array([v for v, m in zip(Exy, mask) if m])
    theta_clean = np.array([v-rota for v, m in zip(theta, mask) if m])
    x_clean = np.array([v for v, m in zip(x_points, mask) if m])
    y_clean = np.array([v for v, m in zip(y_points, mask) if m])
    
    # Rotation matrix --> see https://www.youtube.com/watch?v=-HcDl_gyeMs
    R = np.array([ #deplacement x to x'
    [np.cos(rota), np.sin(rota)],
    [-np.sin(rota),  np.cos(rota)]
    ])
    
    # Prepare arrays for rotated values
    Ux_rot = np.zeros_like(Ux_clean)
    Uy_rot = np.zeros_like(Uy_clean)
    Exx_rot = np.zeros_like(Exx_clean)
    Eyy_rot = np.zeros_like(Eyy_clean)
    Exy_rot = np.zeros_like(Exy_clean)
    
    # Rotate displacement and strain tensors
    for i in range(len(Ux_clean)):
        
        # Rotate displacement
        u = np.array([Ux_clean[i], Uy_clean[i]]) #repere x,y
        u_rot = R @ u
        Ux_rot[i] = u_rot[0] #repere x',y'
        Uy_rot[i] = u_rot[1] #repere x',y'
        
        # Rotate strain tensor
        strain = np.array([ #repere x,y
            [Exx_clean[i], Exy_clean[i]],
            [Exy_clean[i], Eyy_clean[i]]
        ])
        strain_rot = R @ strain @ R.T
        Exx_rot[i] = strain_rot[0, 0] #repere x',y'
        Eyy_rot[i] = strain_rot[1, 1] #repere x',y'
        Exy_rot[i] = strain_rot[0, 1] #repere x',y'
    
    # Rearrange theta into [-π, π] to apply analytical formulae defined within this range
    theta_rearranged = (theta_clean + np.pi) % (2 * np.pi) - np.pi
    
    return Ux_rot,Uy_rot,Exx_rot,Eyy_rot,Exy_rot,theta_rearranged,x_clean,y_clean


def compute_singularity_exponents(
    beta: float,
    ) -> tuple[float,float]:
    
    """
    Main function to compute the eigen values of a sharp V-notch problem for a given notch angle.

    Parameters
    ----------
    beta : float
        V-notch angle (in degrees).
    
    Returns
    -------
    lambda1 : float
        Eigen value solution in mode I.
    lambda2 : float
        Eigen value solution in mode II.
    """    
    
    if beta<0 or beta>np.pi:
        print("Please provide the angle value in rad.")
        return np.NaN,np.NaN
    omega = 2*np.pi-beta
    lmbda1 = np.linspace(0.5,2.,1000000)
    lmbda2 = np.linspace(0.5,2.,1000000)
    fc1 = np.sin(lmbda1*omega)+lmbda1*np.sin(omega)
    fc2 = np.sin(lmbda2*omega)-lmbda2*np.sin(omega)
    if beta==0:
        lambda1=0.5
        lambda2=0.5
    elif beta==np.pi:
        lambda1=1.
        lambda2=2.
    else:
        lamb11,lamb12 = compute_Lamb(lmbda1,fc1)
        lamb21,lamb22 = compute_Lamb(lmbda2,fc2)
        lambda2 = lamb21*(beta<1.79)+lamb22*(beta>=1.79)
        lambda1 = lamb11
    return lambda1,lambda2

def compute_Lamb(
    lmbda: np.ndarray,
    fc: np.ndarray,
    ) -> tuple[float,float]:

    """
    Compute the eigen values by finding the 0.

    Parameters
    ----------
    lmbda : np.ndarray
        Range of possible eigen values varying from 0.5 to 2.
    fc : np.ndarray
        Function to find the 0 for the different eigen values.
    
    Returns
    -------
    lamb1 : float
        First eigen value solution.
    lamb2 : float
        Second eigen value solution.
    """
    
    sgnfc = np.sign(fc)
    sgnfc[np.where(sgnfc==0)]=1
    inds = np.where(np.diff(sgnfc)!=0)[0]
    lamb2 = []
    if len(inds)==1:
        ind1 = inds[0]
        lamb2 = []
    else:
        if len(inds)>=3:
            ind1,ind2 = inds[0:2]
        elif len(inds)==2:
            ind1,ind2 = inds
        a2 = (fc[ind2+1]-fc[ind2])/(lmbda[ind2+1]-lmbda[ind2])
        b2 = fc[ind2+1]-a2*lmbda[ind2+1]
        lamb2 = -b2/a2
    a1 = (fc[ind1+1]-fc[ind1])/(lmbda[ind1+1]-lmbda[ind1])
    b1 = fc[ind1+1]-a1*lmbda[ind1+1]
    lamb1 = -b1/a1
    return lamb1,lamb2

def compute_sigmax_sigmay(
    L1: float,
    L2: float,
    KI: float,
    KII: float,
    radius: float,
    theta_star: float,
    beta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray]:

    """
    Compute asymptotic stress values for a given radius and angle. See paper from Yosibash and Leguillon for a description of the coordinate system.

    Parameters
    ----------
    L1 : float
        Mode I eigen-value associated with the V-notch angle.
    L2 : float
        Mode II eigen-value associated with the V-notch angle.
    KI : float
        GSIF in mode I.
    KII : float
        GSIF in mode II.
    radius : float
        Radius of the point along the contour (scaled).
    theta_star : float
        Angle (theta star) of the point along the contour, see paper from Yoshibash and Leguilon for the coordinate system (in radians).
    beta : float
        V-notch angle (in degrees).
    
    Returns
    -------
    sigma_xx : float
        Stress in the X-direction.
    sigma_yy : float
        Stress in the Y-direction.
    sigma_xy : float
        Shear stress in the cartesian coordinate system.
    sigma_rr : float
        Stress in the radius direction.
    sigma_tt : float
        Stress in the tangential direction.
    sigma_rt : float
        Shear stress in the polar coordinate system.
    """
        
    # Convert deg. to rad.
    beta = beta * np.pi / 180.
        
    # Auxiliary variables
    w = 2*np.pi-beta
        
    theta = theta_star #see paper from Yosibash and Leguillon
    UPL1 = 1+L1
    UML1 = 1-L1
    UPL2 = 1+L2
    UML2 = 1-L2
    TML1 = 3-L1
    TML2 = 3-L2
    sn1wplus  = np.sin(w*UPL1/2)
    sn1wmoins = np.sin(w*UML1/2)
    sn2wplus  = np.sin(w*UPL2/2)
    sn2wmoins = np.sin(w*UML2/2)
        
    # Scaling factors
    S10  = UPL1/UML1*sn1wplus/sn1wmoins-1
    S20  = 1-UML2/UPL2*sn2wplus/sn2wmoins
    
    # Mode I contribution to sigma_rr,sigma_thetatheta and sigma_rtheta
    sigma_rrI  =  np.cos( UPL1*theta_star ) + TML1/UML1 * sn1wplus/sn1wmoins * np.cos( UML1*theta_star )
    sigma_ttI  = -np.cos( UPL1*theta_star ) + UPL1/UML1 * sn1wplus/sn1wmoins * np.cos( UML1*theta_star )
    sigma_rtI  = -np.sin( UPL1*theta_star ) +             sn1wplus/sn1wmoins * np.sin( UML1*theta_star )
    sigma_rrI  = KI * (radius**(L1-1)) * sigma_rrI / S10 
    sigma_ttI  = KI * (radius**(L1-1)) * sigma_ttI / S10
    sigma_rtI  = KI * (radius**(L1-1)) * sigma_rtI / S10
    
    # Mode II contribution to ux and uy
    sigma_rrII =  np.sin( UPL2*theta_star ) + TML2/UPL2 * sn2wplus/sn2wmoins * np.sin( UML2*theta_star )
    sigma_ttII = -np.sin( UPL2*theta_star ) +             sn2wplus/sn2wmoins * np.sin( UML2*theta_star )
    sigma_rtII =  np.cos( UPL2*theta_star ) - UML2/UPL2 * sn2wplus/sn2wmoins * np.cos( UML2*theta_star )
    sigma_rrII = KII * (radius**(L2-1)) * sigma_rrII / S20
    sigma_ttII = KII * (radius**(L2-1)) * sigma_ttII / S20 
    sigma_rtII = KII * (radius**(L2-1)) * sigma_rtII / S20
    
    # Total stress - cylindrical
    sigma_rr = sigma_rrI + sigma_rrII
    sigma_tt = sigma_ttI + sigma_ttII
    sigma_rt = sigma_rtI + sigma_rtII
    
    # Stress rotation from (r,theta) to (x,y)
    cs = np.cos(theta)
    sn = np.sin(theta)
    sigma_xx = cs**2*sigma_rr - 2*sn*cs*sigma_rt + sn**2*sigma_tt
    sigma_yy = sn**2*sigma_rr + 2*sn*cs*sigma_rt + cs**2*sigma_tt
    sigma_xy = sn*cs*(sigma_rr-sigma_tt) + (cs**2-sn**2)*sigma_rt
    	
    return sigma_xx,sigma_yy,sigma_xy,sigma_rr,sigma_tt,sigma_rt


def compute_ux_uy(
    L1: float,
    L2: float,
    KI: float,
    KII: float,
    radius: float,
    theta_star: float,
    Lam: float,
    mu: float,
    beta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray]:
    
    """
    Compute asymptotic displacement values for a given radius and angle. See paper from Yosibash and Leguillon for a description of the coordinate system.

    Parameters
    ----------
    L1 : float
        Mode I eigen-value associated with the V-notch angle.
    L2 : float
        Mode II eigen-value associated with the V-notch angle.
    KI : float
        GSIF in mode I.
    KII : float
        GSIF in mode II.
    radius : float
        Radius of the point along the contour (scaled).
    theta_star : float
        Angle (theta star) of the point along the contour, see paper from Yoshibash and Leguilon for the coordinate system (in radians).
    Lam : float
        First Lamé's coefficient.
    mu : float
        Shear modulus or second Lamé's coefficient.
    beta : float
        V-notch angle (in degrees).
    
    Returns
    -------
    ux : float
        Displacement in the X-direction.
    uy : float
        Displacement in the Y-direction.
    ur : float
        Displacement in the radial direction.
    ut : float
        Displacement in the tangential direction.
    """
    
    # Convert deg. to rad.
    beta = beta * np.pi / 180.
    
    # Auxiliary variables
    w = 2*np.pi-beta
    theta = theta_star #see paper from Yosibash and Leguillon
    LPMU = Lam+mu
    LP3MU = Lam+3*mu
    UPL1 = 1+L1
    UML1 = 1-L1
    UPL2 = 1+L2
    UML2 = 1-L2
    sn1wplus  = np.sin(w*UPL1/2)
    sn1wmoins = np.sin(w*UML1/2)
    sn2wplus  = np.sin(w*UPL2/2)
    sn2wmoins = np.sin(w*UML2/2)
    
    # Scaling factors
    S10  = UPL1/UML1*sn1wplus/sn1wmoins-1 
    S20  = 1-UML2/UPL2*sn2wplus/sn2wmoins 
    
    # Mode I contribution to ux and uy
    urI  = np.cos( UPL1*theta_star ) + (LP3MU-L1*LPMU)/(LPMU*UML1) * sn1wplus/sn1wmoins * np.cos( UML1*theta_star )
    urI  = KI  * (radius**L1) * urI  / ( 2*mu*L1*S10 ) 
    utI  =-np.sin( UPL1*theta_star ) - (LP3MU+L1*LPMU)/(LPMU*UML1) * sn1wplus/sn1wmoins * np.sin( UML1*theta_star ) 
    utI  = KI  * (radius**L1) * utI  / ( 2*mu*L1*S10 )
    
    # Mode II contribution to ux and uy
    urII = np.sin( UPL2*theta_star ) + (LP3MU-L2*LPMU)/(LPMU*UPL2) * sn2wplus/sn2wmoins * np.sin( UML2*theta_star )
    urII = KII * (radius**L2) * urII / ( 2*mu*L2*S20 ) 
    utII = np.cos( UPL2*theta_star ) + (LP3MU+L2*LPMU)/(LPMU*UPL2) * sn2wplus/sn2wmoins * np.cos( UML2*theta_star ) 
    utII = KII * (radius**L2) * utII / ( 2*mu*L2*S20 )
    
    # Total displacement - cylindrical and cartesian
    ur = urI + urII
    ut = utI + utII
    ux = ur*np.cos(theta) - ut*np.sin(theta)
    uy = ur*np.sin(theta) + ut*np.cos(theta)
    
    return ux,uy,ur,ut
    

def compute_u_sigma(
    L1: float,
    L2: float,
    KI: float,
    KII: float,
    r_cir: np.ndarray,
    the: np.ndarray,
    Lam: float,
    mu: float,
    beta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Compute displacement and stress fields for a list of points, for instance along a contour.

    This function calls `compute_ux_uy` and `compute_sigmax_sigmay` for each point defined by its radius and angle (thetas_star) location `(r_cir, the)` and returns both the primal and dual displacement and stress values.

    Parameters
    ----------
    L1, L2 : float
        Eigenvalues associated with the singular solution.
    KI, KII : float
        Mode I and Mode II generalized stress intensity factors.
    r_cir : np.ndarray
        Radius of the points.
    the : np.ndarray
        Angle of the points correspondig to theta_star (in radians).
    Lam : float
        First Lamé's coefficient.
    mu : float
        Shear modulus or second Lamé's coefficient.
    beta : float
        V-notch angle (in degrees).

    Returns
    -------
    ux_p, uy_p, sig1p, sig2p, sig12p : np.ndarray
        Displacement (ux, uy) and stress components (σxx, σyy, σxy) for primal mode.
    ux_d, uy_d, sig1d, sig2d, sig12d : np.ndarray
        Displacement (ux, uy) and stress components (σxx, σyy, σxy) for dual mode.
    """
    
    # Prepare output lists
    ux_p, uy_p, sig1p, sig2p, sig12p = [], [], [], [], []
    ux_d, uy_d, sig1d, sig2d, sig12d = [], [], [], [], []
    
    for rr,theta_star in zip(r_cir,the):
        
        # Displacement and stress - exponent lambda
        uxp,uyp,urp,utp      = compute_ux_uy(L1,L2,KI,KII,rr,theta_star,Lam,mu,beta)
        s1p,s2p,s12p,srrp,sttp,srtp = compute_sigmax_sigmay(L1,L2,KI,KII,rr,theta_star,beta)
        
        # Displacement and stress - exponent -lambda
        uxd,uyd,urd,utd     = compute_ux_uy(-L1,-L2,KI,KII,rr,theta_star,Lam,mu,beta)
        s1d,s2d,s12d,srrd,sttd,srtd = compute_sigmax_sigmay(-L1,-L2,KI,KII,rr,theta_star,beta)
        
        # Append positive exponent results
        ux_p.append(uxp)
        uy_p.append(uyp)
        sig1p.append(s1p)
        sig2p.append(s2p)
        sig12p.append(s12p)

        # Append negative exponent results
        ux_d.append(uxd)
        uy_d.append(uyd)
        sig1d.append(s1d)
        sig2d.append(s2d)
        sig12d.append(s12d)
    
    return np.array(ux_p),np.array(uy_p),np.array(sig1p),np.array(sig2p),np.array(sig12p),np.array(ux_d),np.array(uy_d),np.array(sig1d),np.array(sig2d),np.array(sig12d)


def compute_K(
    center: np.ndarray,
    xcir: np.ndarray,
    ycir: np.ndarray,
    dL: float,
    U1EF: np.ndarray,
    U2EF: np.ndarray,
    S11EF: np.ndarray,
    S22EF: np.ndarray,
    S12EF: np.ndarray,
    ux_p: np.ndarray,
    uy_p: np.ndarray,
    sig1p: np.ndarray,
    sig2p: np.ndarray,
    sig12p: np.ndarray,
    ux_d: np.ndarray,
    uy_d: np.ndarray,
    sig1d: np.ndarray,
    sig2d: np.ndarray,
    sig12d: np.ndarray
    ) -> tuple[float, float, float]:

    """
    Compute the stress intensity factor K and the two associated Psi integrals.
    
    Parameters
    ----------
    center : np.ndarray
        Coordinates of the contour center (x, y), usually it is 0,0.
    xcir, ycir : np.ndarray
        Coordinates of the contour points.
    dL : float
        Arc length increments along the circle (scaled).
    U1EF, U2EF : np.ndarray
        Displacement components from FE/DIC results.
    S11EF, S22EF, S12EF : np.ndarray
        Stress tensor components from FE/DIC results.
    ux_p, uy_p : np.ndarray
        Displacement components for primal field (exponent λ).
    sig1p, sig2p, sig12p : np.ndarray
        Stress tensor components for primal field (exponent λ).
    ux_d, uy_d : np.ndarray
        Displacement components for dual field (exponent -λ).
    sig1d, sig2d, sig12d : np.ndarray
        Stress tensor components for dual field (exponent -λ).
    
    Returns
    -------
    K : float
        Computed stress intensity factor.
    PSI_UEF_Ud : float
        Psi(UEF, r^(-λ)u⁻) integral value.
    PSI_Up_Ud : float
        Psi(r^(λ)u⁺, r^(-λ)u⁻) integral value.
    """
    
    # Compute the normal to the circle points
    normal_X = center[0] - xcir
    normal_Y = center[1] - ycir
    normal_norm = (normal_X**2 + normal_Y**2)**0.5
    normal_X = normal_X / normal_norm
    normal_Y = normal_Y / normal_norm
    
    # Compute sigma(U).n
    sigmaNXEF = S11EF  *normal_X + S12EF  *normal_Y  # Finite element field
    sigmaNYEF = S12EF  *normal_X + S22EF  *normal_Y
    sigmaNXp  = sig1p  *normal_X + sig12p *normal_Y # Primal field - exponent lambda
    sigmaNYp  = sig12p *normal_X + sig2p  *normal_Y
    sigmaNXd  = sig1d  *normal_X + sig12d *normal_Y # Dual field - exponent -lambda
    sigmaNYd  = sig12d *normal_X + sig2d  *normal_Y
    
    # Compute Psi(UEF,r^(-lambda)u-)
    PSI_UEF_Ud = sigmaNXEF*ux_d + sigmaNYEF*uy_d - ( sigmaNXd*U1EF + sigmaNYd*U2EF )
    PSI_UEF_Ud = sum(PSI_UEF_Ud*dL)
    
    # Compute Psi(r^(lambda)u+,r^(-lambda)u-)
    PSI_Up_Ud = sigmaNXp*ux_d   + sigmaNYp*uy_d  - ( sigmaNXd*ux_p + sigmaNYd*uy_p )
    PSI_Up_Ud = sum(PSI_Up_Ud*dL)
    
    # Compute Psi(UEF,r^(-lambda)u-) / Psi(r^(lambda)u+,r^(-lambda)u-)
    K = PSI_UEF_Ud / PSI_Up_Ud
    
    return K,PSI_UEF_Ud,PSI_Up_Ud


def compute_K1K2(
    file: str,
    radius: float,
    center_x: float,
    center_y: float,
    nb_points: int,
    scale: float,
    beta: float,
    E: float,
    nu: float,
    rota: float
    ) -> tuple[float, float]:
    
    """
    Compute the mode I and mode II generalized stress intensity factors (GSIF) from FE/DIC data.
    
    Parameters
    ----------
    file : str
        Path to the DIC .vtk result file.
    radius : float
        Radius of the circular path (in pixels).
    center_x : float
        X-coordinate of the circle center (in pixels).
    center_y : float
        Y-coordinate of the circle center (in pixels).
    nb_points : int
        Number of points along the circle.
    scale : float
        Conversion factor from pixels to millimeters (mm/px).
    beta : float
        V-notch opening angle (in degrees).
    E : float
        Young's modulus (in MPa).
    nu : float
        Poisson's ratio.
    rota : float
        Rotation angle from FE/DIC coordinate system to the one of the V-notch (Yosibach and Leguillon paper) (in radians).
    
    Returns
    -------
    K1 : float
        Mode I GSIF.
    K2 : float
        Mode II GSIF.
    """
    
    # Generate circular points for the contour (in pixels)
    circle_points,theta,dL = generate_circle(center_x,center_y,radius,nb_points)
    
    # Compute DIC displacement and strain (scaled)
    Ux,Uy,Exx,Exy,Eyy,circle = compute_U_E_DIC(file,circle_points,scale)
    
    # Filter points and rotation of coordinate system (FE/DIC to Yosibash and Leguillon's one)
    Ux_DIC,Uy_DIC,Exx_DIC,Eyy_DIC,Exy_DIC,theta_DIC,x_DIC,y_DIC = remove_point_out_rota(Ux,Uy,Exx,Eyy,Exy,theta,circle_points[:,0],circle_points[:,1],rota) #mm
    
    # Compute stress from FE/DIC strain
    Sxx_DIC,Syy_DIC,Sxy_DIC = compute_S_DIC(Exx_DIC,Eyy_DIC,Exy_DIC,E,nu) #MPa
    
    # Material and configuration parameters
    L1,L2 = compute_singularity_exponents(beta*np.pi/180.)
    
    Lam = E*nu / ( (1+nu)*(1-2*nu) )
    mu  = E    / ( 2*(1+nu) )
    r_cir_mm = scale*np.array([radius for aa in theta_DIC]) #mm
    dL_mm = scale * dL #mm
    
    # Convert polar to Cartesian
    x_sur_cercle,y_sur_cercle = r_cir_mm*np.cos(theta_DIC),r_cir_mm*np.sin(theta_DIC)
    
    ###############
    # GSIF mode I #
    ###############

    KI, KII = 1., 0.
    # Compute the displacement and stress fields for Vnotch  #
    ux_p,uy_p,sig1p,sig2p,sig12p,ux_d,uy_d,sig1d,sig2d,sig12d = compute_u_sigma(L1,L2,KI,KII,r_cir_mm,theta_DIC,Lam,mu,beta) #array
    # Compute the contour integral K = PSI(UEF,r^(-lambda)u^-)/PSI(r^(lambda)u+,r^(-lambda)u^-)
    K1,PSI1_UEF_Ud,PSI1_Up_Ud = compute_K([0.,0.],x_sur_cercle,y_sur_cercle,dL_mm,Ux_DIC,Uy_DIC,Sxx_DIC,Syy_DIC,Sxy_DIC,ux_p,uy_p,sig1p,sig2p,sig12p,ux_d,uy_d,sig1d,sig2d,sig12d)
    
    ################
    # GSIF mode II #
    ################

    KI, KII = 0., 1.
    # Compute the displacement and stress fields for Vnotch  #
    ux_p,uy_p,sig1p,sig2p,sig12p,ux_d,uy_d,sig1d,sig2d,sig12d = compute_u_sigma(L1,L2,KI,KII,r_cir_mm,theta_DIC,Lam,mu,beta) #array
    # Compute the contour integral K = PSI(UEF,r^(-lambda)u^-)/PSI(r^(lambda)u+,r^(-lambda)u^-)
    K2,PSI2_UEF_Ud,PSI2_Up_Ud = compute_K([0.,0.],x_sur_cercle,y_sur_cercle,dL_mm,Ux_DIC,Uy_DIC,Sxx_DIC,Syy_DIC,Sxy_DIC,ux_p,uy_p,sig1p,sig2p,sig12p,ux_d,uy_d,sig1d,sig2d,sig12d)

    return K1,K2
