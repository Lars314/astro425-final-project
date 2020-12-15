import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt


class Lens:
    """
    """
    def draw(self, do_circle=True, do_log_scale=False):
        """Plot the lensed and unlensed images"""
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(20, 10)

        ax[0].tick_params(axis='both', which='both', labelsize=20)
        ax[1].tick_params(axis='both', which='both', labelsize=20)

        extent_scale = [(-self.shape[1] * self.pixel_size.value) / 2,
                        (self.shape[1] * self.pixel_size.value) / 2,
                        (-self.shape[0] * self.pixel_size.value) / 2,
                        (self.shape[0] * self.pixel_size.value) / 2]

        ax[0].imshow(self.source, extent=extent_scale)
        ax[1].imshow(self.result, extent=extent_scale)

        ax[0].set_title("Source Image", fontsize=30)
        ax[1].set_title("Observed Image", fontsize=30)

        ax[0].set_xlabel(r"$\theta$" + "   [{0}]".format(
            self.pixel_size.unit.to_string()), fontsize=24)
        ax[1].set_xlabel(r"$\theta$" + "   [{0}]".format(
            self.pixel_size.unit.to_string()), fontsize=24)
        ax[0].set_ylabel(r"$\theta$" + "   [{0}]".format(
            self.pixel_size.unit.to_string()), fontsize=24)
        ax[1].set_ylabel(r"$\theta$" + "   [{0}]".format(
            self.pixel_size.unit.to_string()), fontsize=24)

        if (do_circle):
            einstein_ring = self.einstein_radius.to('arcsec').value
            ring1 = plt.Circle((self.lens_x, self.lens_y),
                               einstein_ring, color='w',
                               linewidth=3, fill=False)
            ring0 = plt.Circle((self.lens_x, self.lens_y),
                               einstein_ring, color='w',
                               linewidth=3, fill=False)
            ax[0].add_artist(ring0)
            ax[1].add_artist(ring1)
            ax[0].plot(self.lens_x, self.lens_y,
                       color='w', marker='o', markersize=10)
            ax[1].plot(self.lens_x, self.lens_y,
                       color='w', marker='o', markersize=10)

        fig.tight_layout()

    def find_theta_(self, beta):
        """
        """
        theta_p = \
            (1/2)*(beta + np.sqrt((beta**2) + 4*(self.einstein_radius**2)))

        theta_m = \
            (1/2)*(beta - np.sqrt((beta**2) + 4*(self.einstein_radius**2)))
        return theta_p, theta_m

    def find_beta_(self, theta):
        """
        """
        beta = ((theta**2) - (self.einstein_radius**2)) / theta

        return beta

    def deflect(self):
        """
        """
        observed = np.zeros(shape=self.shape)

        # loop through the source image
        for j in range(0, self.shape[0]):
            if ((self.verbosity >= 1) and (j % (self.shape[0] / 10) == 0)):
                print("now at j : {0}".format(j))
            for i in range(0, self.shape[1]):
                # this pixel is empty, we can skip it for efficiency
                if (self.source[j][i] == 'bruh'):
                    continue
                else:
                    # it is easier to think about if we put the lens at 0,0
                    # so we need to correct our coordinates from array indices
                    # into cartesian coordinates
                    x = (i - self.lens_i) * self.pixel_size
                    y = -(j - self.lens_j) * self.pixel_size

                    # this gives us an error when calculating phi, the x=0 and
                    # x!=0 cases need to be handled separately
                    if ((x == 0) or (y == 0)):
                        continue
                        # phi = np.pi / 2
                    else:
                        phi = np.arctan(y/x)

                    # now we can find our beta and theta. Because we adjusted
                    # x and y with self.pixel_size, which is an angle, we treat
                    # x and y as angles in the sky and can get beta right away
                    theta = np.sqrt((x**2) + (y**2))

                    # because we are converting between cartesian and radial
                    # coordinates, we need a correction for signs. Particularly
                    # because cos(-phi) = cos(+phi), but we need to make sure
                    # our mapping has the right directionality
                    # if (x/y < 0):
                    #    theta = -1 * theta

                    beta = self.find_beta_(theta)

                    # now we use our theta and phi to get the x y coordinates.
                    # we are just converting from radial to cartesian here, but
                    # because our coordinates are angles in the sky, we use
                    # theta as the radial coordinate

                    c = -1
                    if ((x < 0) or ((x == 0) and (y < 0))):
                        c = 1

                    new_x_p = c * beta * np.cos(phi)
                    new_y_p = c * beta * np.sin(phi)

                    # we need our magnification corrections, so as not to have
                    # disproportionate brightness. Particularly the negative
                    # solution will dominate when uncorrected, and the object
                    # is outside the einstein radius
                    u = np.abs(beta) / self.einstein_radius
                    if (u == 0):
                        continue
                    if ((theta // beta) >= 0):
                        # we are dealing with the positive solution
                        mag = (2+u**2)/(2*u*np.sqrt(4+u**2)) + 0.5
                    elif ((theta // beta) <= 0):
                        mag = (2+u**2)/(2*u*np.sqrt(4+u**2)) - 0.5

                    # we need integer numpy array coordinates. Using numpy's
                    # round function we can reduce artifacting, by not always
                    # just rounding down
                    i_p = int(-np.round(new_x_p / self.pixel_size)) \
                        + self.lens_i
                    j_p = int(-np.round(-new_y_p / self.pixel_size)) \
                        + self.lens_j

                    if (self.verbosity >= 2):
                        print("i : {0}".format(i))
                        print("j : {0}".format(j))
                        print("x : {0}".format(x))
                        print("y : {0}".format(y))
                        print("theta : {0}".format(theta))
                        print("beta : {0}".format(beta))
                        print("phi : {0}".format(phi))
                        print("new x p : {0}".format(new_x_p))
                        print("new y p : {0}".format(new_y_p))
                        print("i p : {0}".format(i_p))
                        print("j p : {0}".format(j_p))
                        print("mag p : {0}".format(mag))

                    # now we can map values from the source to the observed
                    # image. Particularly with complex source images we might
                    # get indices outside the dimensions of the observed image.
                    # if this happens, we just skip those
                    if((j_p >= 0) and (i_p >= 0)):
                        try:
                            observed[j][i] += self.source[j_p][i_p] * mag
                        except(IndexError):
                            continue

        return observed

    def __init__(self,
                 source_image,
                 lensLoc,
                 pixel_size,
                 mass=(10**13)*u.M_sun,
                 lens_dist=1.4 * u.Gpc,
                 source_dist=1.73 * u.Gpc,
                 ls_dist=1.41 * u.Gpc,
                 verbosity=0):

        self.verbosity = verbosity

        # Note for future me! We need this in radians for some math later on!
        self.einstein_radius = np.sqrt((4*const.G*mass*ls_dist)/(
            source_dist*lens_dist*(const.c**2))) * u.rad

        self.pixel_size = pixel_size
        self.source = source_image
        self.shape = source_image.shape

        # the indices here are frustrating, shape uses (y, x) but I am using
        # (x, y) in the lensLoc parameter. lensLoc is what interacts with the
        # user, so it is better to keep standard notation there
        self.lens_x = lensLoc[0]
        self.lens_y = lensLoc[1]

        self.lens_i = int((lensLoc[0] // pixel_size.value)
                          + (self.shape[1] // 2))
        # the negative here is to correct for flipped y, numpy indices have
        # y increase going down, but our user will want to work with increasing
        # y going up
        self.lens_j = int((-lensLoc[1] // pixel_size.value)
                          + (self.shape[0] // 2))

        self.result = self.deflect()
