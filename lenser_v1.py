import numpy as np
import astropy.units as u
import astropy.constants as const


class Plane:
    """
    Defines a plane object
    """
    def __init__(self, image, units, dist):
        self.image = image
        self.units = units
        self.dist = dist


class Lens:
    """
    """
    def find_theta_(self, beta):
        """
        """
        theta_plus = \
            (1/2)*(beta + np.sqrt((beta**2) + 4*(self.theta_einstein**2)))

        theta_minus = \
            (1/2)*(beta - np.sqrt((beta**2) + 4*(self.theta_einstein**2)))
        return theta_plus, theta_minus

    def __init__(self, mass,
                 lens_dist, source_dist, ls_dist,
                 source_image, lens_loc, verbosity):
        """
        mass : [mass] the mass of the lens

        lens_dist : [distance] the distance from the observer to the lens

        source_dist : [distance] the distance from the oberser to the source
        """
        self.mass = mass
        self.theta_einstein = np.sqrt((4*const.G*mass*ls_dist)/(
            source_dist*lens_dist*(const.c**2))) * u.rad

        o_x = lens_loc[0]
        o_y = lens_loc[1]

        # lens_image = np.zeros(shape=(1000, 1000))
        # lens_plane = Plane(lens_image, 1*u.arcsec, lens_dist)
        source_plane = Plane(source_image, 1*u.arcsec, source_dist)

        observed_image = np.zeros(shape=source_image.shape)

        # go through each pixel in the source plane
        for y in range(1, len(source_plane.image)):
            this_y = (y-o_y) * source_plane.units
            for x in range(1, len(source_plane.image[y])):
                # x = x - (len(source_plane.image) / 2)
                # y = y - (len(source_plane.image) / 2)
                this_x = (x-o_x) * source_plane.units

                if (verbosity):
                    print("y = {0}".format(y))
                    print("x = {0}".format(x))
                    print("this y = {0}".format(this_y))
                    print("this x = {0}".format(this_x))

                r = np.sqrt((this_x**2) + (this_y**2))

                if(this_x == 0 or this_y == 0):
                    continue
                phi = np.arctan(this_y/this_x)

                if (verbosity):
                    print("r = {0}".format(r))
                    print("phi = {0}".format(phi))

                # B = np.arctan(r.value) * u.arcsec
                B = r
                theta_plus, theta_minus = self.find_theta_(B)

                if (verbosity):
                    print("beta = {0}".format(B))
                    print("theta + = {0}".format(theta_plus))
                    print("theta - = {0}".format(theta_minus))

                # new_r_plus = source_dist * np.tan(theta_plus)
                new_r_plus = theta_plus
                
                if (this_x > 0):
                    new_x_plus = int(new_r_plus.value * np.cos(phi))
                    new_y_plus = int(new_r_plus.value * np.sin(phi))
                if (this_x < 0):
                    new_x_plus = -int(new_r_plus.value * np.cos(phi))
                    new_y_plus = -int(new_r_plus.value * np.sin(phi))

                new_r_minus = theta_minus
                new_y_minus = int(new_r_minus.value * np.sin(phi))
                new_x_minus = int(new_r_minus.value * np.cos(phi))

                if (verbosity):
                    print("new r + = {0}".format(new_r_plus))
                    print("new y + = {0}".format(new_y_plus))
                    print("new x + = {0}".format(new_x_plus))
                    print("new y - = {0}".format(new_y_minus))
                    print("new x - = {0}".format(new_x_minus))

                y_coord_plus = o_y + new_y_plus
                x_coord_plus = o_x + new_x_plus
                y_coord_minus = o_y + new_y_minus
                x_coord_minus = o_x + new_x_minus

                """
                if ((this_x.value > 0.0) and (this_y.value > 0.0)):
                    # Q4
                    y_coord_plus = o_y + new_y_plus
                    x_coord_plus = o_x + new_x_plus
                if ((this_x.value < 0.0) and (this_y.value > 0.0)):
                    # Q3
                    y_coord_plus = o_y + new_y_plus
                    x_coord_plus = o_x - new_x_plus
                if ((this_x.value < 0.0) and (this_y.value < 0.0)):
                    # Q2
                    y_coord_plus = o_y - new_y_plus
                    x_coord_plus = o_x - new_x_plus

                if ((this_x.value > 0.0) and (this_y.value < 0.0)):
                    # Q1
                    y_coord_plus = o_y + new_y_plus
                    x_coord_plus = o_x + new_x_plus

                if (verbosity):
                    print("y_coord + = {0}".format(y_coord_plus))
                    print("x_coord + = {0}".format(x_coord_plus))
                """
                if ((x_coord_plus > 0) and (y_coord_plus > 0)):
                    try:
                        observed_image[y_coord_plus][x_coord_plus] += \
                            source_plane.image[y][x]
                        # observed_image[y_coord_minus][x_coord_minus] += \
                        #    source_plane.image[y][x]
                    except:
                        continue

                if (verbosity):
                    print("--------------------")

        self.result = observed_image
