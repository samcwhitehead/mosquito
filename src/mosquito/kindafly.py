"""
Code to generate a user-adjustable frame that can be overlaid on videos (high-speed and normal)
for kinematics extraction. Taking heavily from Kinefly, but trying to keep it simpler

TODO:
    - parent/child stuff for chained transformations
    - try to reduce redundancy in set_params/handles

"""

# ---------------------------------------
# IMPORTS
# ---------------------------------------
import cv2

import numpy as np

# ---------------------------------------
# PARAMS
# ---------------------------------------
# dictionary for converting color strings to bgra values
bgra_dict = dict(black=(0.0, 0.0, 0.0, 0.0), white=(255.0, 255.0, 255.0, 0.0), dark_gray=(64.0, 64.0, 64.0, 0.0),
                 gray=(128.0, 128.0, 128.0, 0.0), light_gray=(192.0, 192.0, 192.0, 0.0), red=(0.0, 0.0, 255.0, 0.0),
                 green=(0.0, 255.0, 0.0, 0.0), blue=(255.0, 0.0, 0.0, 0.0), cyan=(255.0, 255.0, 0.0, 0.0),
                 magenta=(255.0, 0.0, 255.0, 0.0), yellow=(0.0, 255.0, 255.0, 0.0), dark_red=(0.0, 0.0, 128.0, 0.0),
                 dark_green=(0.0, 128.0, 0.0, 0.0), dark_blue=(128.0, 0.0, 0.0, 0.0),
                 dark_cyan=(128.0, 128.0, 0.0, 0.0), dark_magenta=(128.0, 0.0, 128.0, 0.0),
                 dark_yellow=(0.0, 128.0, 128.0, 0.0), light_red=(175.0, 175.0, 255.0, 0.0),
                 light_green=(175.0, 255.0, 175.0, 0.0), light_blue=(255.0, 175.0, 175.0, 0.0),
                 light_cyan=(255.0, 255.0, 175.0, 0.0), light_magenta=(255.0, 175.0, 255.0, 0.0),
                 light_yellow=(175.0, 255.0, 255.0, 0.0))

# dictionary giving default params
default_params = dict(
    body_axis=dict(
        start_pt=np.array([400, 320]),
        end_pt=np.array([80, 320])
    ),
    left_wing=dict(
        hinge_pt=np.array([300, 40+315]),  # np.array([0, 40]),
        radius_inner=80,
        radius_outer=120,
        angle_hi=2.3562,
        angle_lo=0.7854,
    ),
    right_wing=dict(
        hinge_pt=np.array([300, -40+315]),  # np.array([0, -40]),
        radius_inner=80,
        radius_outer=120,
        angle_hi=-0.7854,
        angle_lo=-2.3562,
    ),
                      )


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
def get_angle_from_points(pt1, pt2):
    """
    Convenience funtion for getting angle from 2D points
    """
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]

    return np.arctan2(y, x)


def angle_between_vectors(u, v):
    """
    Convenience funtion for getting angle BETWEEN two vectors
    """
    theta = np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    sign_ = np.sign(np.cross(u, v))

    return sign_*theta


def mat_from_angle_and_trans(angle, translation):
    """
    Convenience function to return a 2D transformation matrix given
    an input angle and translation

    """
    mat = np.array([[np.cos(angle), -1 * np.sin(angle), translation[0]],
                    [np.sin(angle), np.cos(angle), translation[1]],
                    [0, 0, 1]])
    return mat


def apply_transform(pt, transform):
    """
    Convenience function to apply a 2D affine transformation matrix to a point
    and return the result (saves me from having to mess with homogeneous
    coordinates in main code

    """
    pt_homo = np.pad(pt, (0, 1), 'constant', constant_values=(0, 1))
    return (transform @ pt_homo)[:2]


def cart2pol(x):
    """
    Convenience function for cartesian to polar coordinates
    """
    rho = np.linalg.norm(x)
    phi = np.arctan2(x[1], x[0])
    return rho, phi


def pol2cart(rho, phi):
    """
    Convenience function for polar to cartesian coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])


def get_body_angle(start_pt, end_pt):
    """
    Convenience function to calculate body angle
    (makes sure I'm always doing it the same way)

    thorax = start_pt
    head = end_pt
    """
    theta_b = get_angle_from_points(start_pt, end_pt)
    # unwrap to [0, 2*pi]
    if theta_b < 0:
        theta_b += 2*np.pi
    return theta_b


# ---------------------------------------
# UI CLASSES
# ---------------------------------------
# ------------------------------------------------------------------------------------------
# CLASS TO CREATE A MOVABLE POINT HANDLE (FROM KINEFLY)
# ------------------------------------------------------------------------------------------
class Handle(object):
    """
    Handle class from Kinefly
    """
    def __init__(self, pt=np.array([0, 0]), color=(255.0, 255.0, 255.0, 0.0), name=None):
        self.pt = pt
        self.name = name
        self.scale = 1.0

        self.color = color
        self.radiusDraw = 3
        self.radiusHit = 6

    def hit_test(self, pt_mouse):
        d = np.linalg.norm(self.pt - pt_mouse)
        if d < self.radiusHit:
            return True
        else:
            return False

    def draw(self, image):
        cv_filled = -1
        cv2.circle(image, tuple(self.pt.astype(int)), self.radiusDraw,
                   self.color, cv_filled)


# ------------------------------------------------------------------------------------------
# CLASS TO DRAW AN ADJUSTABLE LINE (EG FOR BODY AXIS)
# ------------------------------------------------------------------------------------------
class MyLine(object):
    """
    Line class adapted from Kinefly
    """

    def __init__(self, name=None, color='white', params=default_params, followers=[]):

        self.name = name

        self.color = color
        self.color_bgra = bgra_dict[color]

        # UI handles (points that can be clicked and dragged)
        self.params = params
        start_pt = self.params[self.name]['start_pt']
        end_pt = self.params[self.name]['end_pt']
        self.handles = {'start_pt': Handle(start_pt, bgra_dict['yellow'], name='start_pt'),
                        'end_pt': Handle(end_pt, bgra_dict['magenta'], name='end_pt')}

        # followers refer to objects that should transform with this one
        self.followers = followers

        # other variables
        self.mask = None  # these aren't used, but are present to keep class variables consistent
        self.roi = None

    # update handle points
    def update_handles_from_params(self):
        self.handles['start_pt'].pt = self.params[self.name]['start_pt']
        self.handles['end_pt'].pt = self.params[self.name]['end_pt']

    # update params, as from changes in value to handles via mouse input
    def update_params_from_handles(self):
        # read out current copy of params
        params = self.params

        # update param values based on inputs
        params[self.name]['start_pt'] = self.handles['start_pt'].pt
        params[self.name]['end_pt'] = self.handles['end_pt'].pt

        # set parameters
        self.params = params
        # self.set_params(params)

    # update handles based on input e.g. from mouse movement
    def update_handles_from_input(self, tag, new_val):
        self.handles[tag].pt = new_val

    # Make an empty method for get_mask()
    def get_mask(self, image_shape):
        pass

    # Draw all handle points
    def draw_handles(self, image):
        for tagHandle, handle in self.handles.items():
            handle.draw(image)

    # Draw the line connecting the handles + handles themselves
    def draw(self, image):
        # draw handle points
        self.draw_handles(image)

        # draw line
        start_pt = self.params[self.name]['start_pt']
        end_pt = self.params[self.name]['end_pt']
        cv2.line(image, start_pt, end_pt, self.color_bgra, 1)

    # Get the UI object, if any, that the mouse is on.
    def hit_object(self, pt_mouse):
        tag = None

        # Check for handle hits.
        for tag_handle, handle in self.handles.items():
            if handle.hit_test(pt_mouse):
                tag = tag_handle
                break

        return self.name, tag

    # Define the transformation going from parent frame to child (this) frame
    def parent_to_self_transform(self):
        start_pt = self.params[self.name]['start_pt']
        end_pt = self.params[self.name]['end_pt']
        angle = get_angle_from_points(start_pt, end_pt)
        translation = start_pt
        return mat_from_angle_and_trans(angle, translation)

    # # Define the transformation going from global frame to child (this) frame
    # def global_to_child_transform(self):
    #     if self.parent is None:
    #         mat, angle, translation = self.parent_to_child_transform()


# ------------------------------------------------------------------------------------------
# CLASS TO DRAW AN ADJUSTABLE WEDGE (EG FOR WING TRACKING)
# ------------------------------------------------------------------------------------------
class MyWedge(object):
    def __init__(self, parent=None, name=None, color='white', params=default_params, followers=[]):
        self.parent = parent

        self.name = name

        self.color = color
        self.color_bgra = bgra_dict[color]
        self.color_bgra_dim = tuple(0.8*np.array(self.color_bgra))

        self.params = params

        hinge_pt = self.params[self.name]['hinge_pt']
        self.handles = {'hinge_pt': Handle(hinge_pt, self.color_bgra, name='hinge_pt'),
                        'angle_hi': Handle(np.array([0, 0]), self.color_bgra, name='angle_hi'),
                        'angle_lo': Handle(np.array([0, 0]), self.color_bgra, name='angle_lo'),
                        'radius_inner': Handle(np.array([0, 0]), self.color_bgra, name='radius_inner')}

        # followers refer to objects that should transform with this one
        self.followers = followers

        # other variables
        self.body_angle = None
        self.angle_hi_i = None
        self.angle_lo_i = None
        self.parent2self = None
        self.self2parent = None
        self.update_hi_lo_angles()

        self.pt_wedge_hi_outer = None  # these are the points to draw lines bounding the wedge
        self.pt_wedge_hi_inner = None
        self.pt_wedge_lo_outer = None
        self.pt_wedge_lo_inner = None

        self.mask = None
        self.roi = None

        # do initial handle update
        self.update_handles_from_params()

    def update_hi_lo_angles(self):
        self.update_body_angle()
        self.angle_hi_i = self.params[self.name]['angle_hi']  # - self.body_angle
        self.angle_lo_i = self.params[self.name]['angle_lo']  # - self.body_angle

    # convenience function to update wedge outer boundary lines
    def update_wedge_points_from_params(self):
        # get current values of handle points
        hinge_pt = self.params[self.name]['hinge_pt']
        radius_outer = self.params[self.name]['radius_outer']
        radius_inner = self.params[self.name]['radius_inner']
        angle_hi_i = self.angle_hi_i
        angle_lo_i = self.angle_lo_i

        # # TEMP transform wedge points to image frame
        # hinge_pt = apply_transform(hinge_pt, self.self2parent)

        # set wedge points
        self.pt_wedge_hi_outer = hinge_pt + (radius_outer * np.array([np.cos(angle_hi_i), np.sin(angle_hi_i)]))
        self.pt_wedge_hi_inner = hinge_pt + (radius_inner * np.array([np.cos(angle_hi_i), np.sin(angle_hi_i)]))
        self.pt_wedge_lo_outer = hinge_pt + (radius_outer * np.array([np.cos(angle_lo_i), np.sin(angle_lo_i)]))
        self.pt_wedge_lo_inner = hinge_pt + (radius_inner * np.array([np.cos(angle_lo_i), np.sin(angle_lo_i)]))

    def update_handles_from_params(self):
        # read values from params
        hinge_pt = self.params[self.name]['hinge_pt']
        radius_outer = self.params[self.name]['radius_outer']
        radius_inner = self.params[self.name]['radius_inner']
        # self.update_hi_lo_angles()
        angle_hi_i = self.angle_hi_i
        angle_lo_i = self.angle_lo_i
        angle = (angle_hi_i + angle_lo_i) / 2.0

        # # TEMP convert to image frame
        # hinge_pt = apply_transform(hinge_pt, self.self2parent)

        self.handles['hinge_pt'].pt = hinge_pt
        self.handles['radius_inner'].pt = hinge_pt + (radius_inner * np.array([np.cos(angle), np.sin(angle)]))
        self.handles['angle_hi'].pt = hinge_pt + (radius_outer * np.array([np.cos(angle_hi_i), np.sin(angle_hi_i)]))
        self.handles['angle_lo'].pt = hinge_pt + (radius_outer * np.array([np.cos(angle_lo_i), np.sin(angle_lo_i)]))

        # also update wedge boundary lines
        self.update_wedge_points_from_params()

    def update_params_from_handles(self):
        # read out current copy of params
        params = self.params

        # get angle of longitudinal body axis in plane
        self.update_body_angle()

        # get current values of handle points
        hinge_pt = self.handles['hinge_pt'].pt
        radius_inner_pt = self.handles['radius_inner'].pt
        angle_hi_pt = self.handles['angle_hi'].pt
        angle_lo_pt = self.handles['angle_lo'].pt

        # # TEMP transform to body frame
        # hinge_pt = apply_transform(hinge_pt, self.parent2self)  # TEMP go image to body frame
        # radius_inner_pt = apply_transform(radius_inner_pt, self.parent2self)  # TEMP go image to body frame
        # angle_hi_pt = apply_transform(angle_hi_pt, self.parent2self)  # TEMP go image to body frame
        # angle_lo_pt = apply_transform(angle_lo_pt, self.parent2self)  # TEMP go image to body frame

        # use these to define the param values
        params[self.name]['hinge_pt'] = hinge_pt
        params[self.name]['radius_inner'] = np.linalg.norm(radius_inner_pt - hinge_pt)
        params[self.name]['radius_outer'] = np.linalg.norm(angle_hi_pt - hinge_pt)
        params[self.name]['angle_hi'] = get_angle_from_points(hinge_pt, angle_hi_pt)  #  - self.body_angle
        params[self.name]['angle_lo'] = get_angle_from_points(hinge_pt, angle_lo_pt)  #  - self.body_angle

        # set parameters
        self.params = params
        self.update_hi_lo_angles()

        # also update wedge boundary lines
        self.update_wedge_points_from_params()

    # update handles based on e.g. mouse input
    def update_handles_from_input(self, tag, new_val):
        # how has the tagged point moved?
        move_vec = new_val - self.handles[tag].pt

        # how we move handles depends on their type
        if tag == 'hinge_pt':
            # if the hinge moves, move everything with it
            self.handles['hinge_pt'].pt += move_vec
            self.handles['radius_inner'].pt += move_vec
            self.handles['angle_hi'].pt += move_vec
            self.handles['angle_lo'].pt += move_vec

        elif tag == 'radius_inner':
            # if the inner radius changes, move the inner contour of the wedge
            # (bounded by outer radius and fixed to center of arc)
            move_vec = new_val - self.handles[tag].pt
            radius_inner_p = self.handles['radius_inner'].pt - self.handles['hinge_pt'].pt  # in frame of the hinge
            move_vec_proj = (np.dot(move_vec, radius_inner_p)/np.dot(radius_inner_p, radius_inner_p))*radius_inner_p
            radius_inner_new = move_vec_proj + self.handles['radius_inner'].pt  # where to move in

            # check that this isn't beyond the outer radius
            radius_inner_new_p = radius_inner_new - self.handles['hinge_pt'].pt  # inner radius in part frame
            if np.linalg.norm(radius_inner_new_p) > self.params[self.name]['radius_outer']:
                scale = (self.params[self.name]['radius_outer'] - 2)/np.linalg.norm(radius_inner_new_p)
                radius_inner_new = scale*radius_inner_new_p + self.handles['hinge_pt'].pt

            # update handle
            self.handles['radius_inner'].pt = radius_inner_new

        elif tag in ['angle_hi', 'angle_lo']:
            # try this one in polar coordinates
            new_val_rho, new_val_theta = cart2pol(new_val - self.handles['hinge_pt'].pt)
            old_val_rho, old_val_theta = cart2pol(self.handles[tag].pt - self.handles['hinge_pt'].pt)
            delta_rho = new_val_rho - old_val_rho
            delta_theta = new_val_theta - old_val_theta

            # bound radial change by inner radius
            delta_rho = max(delta_rho,
                            self.params[self.name]['radius_inner'] - self.params[self.name]['radius_outer'] + 2)

            # loop over angle vals
            for angle_tag in ['angle_hi', 'angle_lo', 'radius_inner']:
                # get polar representation of current vector
                rho, theta = cart2pol(self.handles[angle_tag].pt - self.handles['hinge_pt'].pt)

                # apply changes in polar coords
                if not angle_tag == 'radius_inner':
                    rho += delta_rho

                if angle_tag == tag:
                    theta += delta_theta
                elif angle_tag == 'radius_inner':
                    theta += delta_theta/2

                # return to cartesian and update
                self.handles[angle_tag].pt = pol2cart(rho, theta) + self.handles['hinge_pt'].pt

    # get body axis angle
    def update_body_angle(self):
        start_pt = self.params['body_axis']['start_pt']  # thorax
        end_pt = self.params['body_axis']['end_pt']  # head
        self.body_angle = get_body_angle(start_pt, end_pt)  # + np.pi

        # also get transformation (TEMP leaving out translation for now)
        self.parent2self = mat_from_angle_and_trans(-1*self.body_angle, -1*start_pt)
        self.self2parent = mat_from_angle_and_trans(self.body_angle, start_pt)

    # Get a mask corresponding to the wedge region
    def get_mask(self, image_shape):
        # initialize image
        mask = np.zeros(image_shape)

        # read out values for handle points
        hinge_pt = self.params[self.name]['hinge_pt'].astype('int')
        # hinge_pt = apply_transform(hinge_pt, self.parent2self)  # TEMP
        radius_outer = int(self.params[self.name]['radius_outer'])
        radius_inner = int(self.params[self.name]['radius_inner'])
        angle_hi = np.rad2deg(self.angle_hi_i)
        angle_lo = np.rad2deg(self.angle_lo_i)

        # draw a white filled circle at outer radius (in angle range)
        cv2.ellipse(mask,
                    hinge_pt,
                    (radius_outer, radius_outer),
                    0,
                    angle_hi,
                    angle_lo,
                    bgra_dict['white'],
                    -1)

        # draw a black filled circle at inner radius
        cv2.ellipse(mask,
                    hinge_pt,
                    (radius_inner, radius_inner),
                    0,
                    0,
                    360,
                    bgra_dict['black'],
                    -1)

        # Make the mask one pixel bigger to account for pixel aliasing.
        mask = cv2.dilate(mask, np.ones([3, 3]))
        self.mask = mask

        # get roi for mask?
        x_sum = np.sum(mask, 0)
        y_sum = np.sum(mask, 1)
        x_list = np.where(x_sum > 0)[0]
        y_list = np.where(y_sum > 0)[0]

        if (len(x_list)>0) and (len(y_list)>0):
            xmin = x_list[0]
            xmax = x_list[-1] + 1
            ymin = y_list[0]
            ymax = y_list[-1] + 1
        else:
            xmin = None
            xmax = None
            ymin = None
            ymax = None

        self.roi = np.array([xmin, ymin, xmax, ymax])

    # Draw all handle points
    def draw_handles(self, image):
        for tagHandle, handle in self.handles.items():
            handle.draw(image)

    # Draw the wedge
    def draw(self, image):
        # draw handle points
        self.draw_handles(image)

        # read out values for drawing
        hinge_pt = self.params[self.name]['hinge_pt'].astype('int')

        # # TEMP
        # hinge_pt = self.params[self.name]['hinge_pt']
        # hinge_pt = apply_transform(hinge_pt, self.self2parent).astype('int')
        # # print(hinge_pt)

        radius_outer = int(self.params[self.name]['radius_outer'])
        radius_inner = int(self.params[self.name]['radius_inner'])
        radius_mid = int((radius_outer + radius_inner)/2)
        angle_hi = np.rad2deg(self.angle_hi_i)
        angle_lo = np.rad2deg(self.angle_lo_i)

        # Draw the arcs
        for radius in [radius_inner, radius_mid, radius_outer]:
            cv2.ellipse(image,
                        hinge_pt,
                        (radius, radius),
                        0,
                        angle_hi,
                        angle_lo,
                        self.color_bgra_dim,
                        1)

        # Draw wedge lines
        cv2.line(image,
                 tuple(self.pt_wedge_hi_inner.astype('int')),
                 tuple(self.pt_wedge_hi_outer.astype('int')),
                 self.color_bgra_dim,
                 1)
        cv2.line(image,
                 tuple(self.pt_wedge_lo_inner.astype('int')),
                 tuple(self.pt_wedge_lo_outer.astype('int')),
                 self.color_bgra_dim,
                 1)

    # Get the UI object, if any, that the mouse is on.
    def hit_object(self, pt_mouse):
        tag = None

        # Check for handle hits.
        for tagHandle, handle in self.handles.items():
            if handle.hit_test(pt_mouse):
                tag = tagHandle
                break

        return self.name, tag


# ------------------------------------------------------------------------------------------
# CLASS FOR A COMPILED FLY FRAME TO ALLOW MULTIPLE BODY PART TRACKING
# ------------------------------------------------------------------------------------------
class FlyFrame:
    def __init__(self, name='fly frame', params=default_params, init_dict=None):
        # want to have the option to initialize from an existing fly_frame, which we'll
        # typically save as a dict (fly_frame.__dict__). otherwise, use defaults
        if init_dict is not None:
            self.__dict__ = init_dict

        else:
            # read out name
            self.name = name

            # set params
            self.params = params

            # generate a set of objects
            self.body_axis = MyLine(name='body_axis', params=self.params,
                                    followers=['right_wing', 'left_wing'])
            self.right_wing = MyWedge(name='right_wing', params=self.params, color='red')
            self.left_wing = MyWedge(name='left_wing', params=self.params, color='green')
            self.partnames = ['body_axis', 'right_wing', 'left_wing']

            # values for current state of system
            self.mousing = False
            self.current_tag = None
            self.current_partname = None
            self.current_part = None

            # current image in system
            self.image_shape = (640, 640)  # NB: image shape needs to be in reverse order from np arrays

    # convenience function to get long body axis angle
    def get_body_angle(self):
        start_pt = self.params['body_axis']['start_pt']
        end_pt = self.params['body_axis']['end_pt']
        return get_body_angle(start_pt, end_pt)

    # which object is being clicked on
    def hit_object(self, pt_mouse):
        tag_hit = None
        partname_hit = None

        for partname in self.partnames:
            name, tag = getattr(self, partname).hit_object(pt_mouse)
            if tag is not None:
                tag_hit = tag
                partname_hit = name
                break

        return partname_hit, tag_hit

    # define click/drag callback
    def on_mouse(self, event, x, y, flags, param):
        # get mouse click point clipped to image
        pt_mouse = np.array([x, y]).clip((0, 0), self.image_shape)

        # see if we have a UI element selected during downclick
        if event == cv2.EVENT_LBUTTONDOWN:

            # Get the name and ui nearest the current point.
            partname, tag = self.hit_object(pt_mouse)
            self.current_tag = tag
            self.current_partname = partname
            if tag is not None:
                self.current_part = getattr(self, partname)
            self.mousing = True

        # during upclick, release any selected UI element
        elif event == cv2.EVENT_LBUTTONUP:
            self.current_tag = None
            self.current_partname = None
            self.current_part = None
            self.mousing = False

        # update coordinates based on mouse movement
        # (if we have a tagged element selected, this will move it)
        if self.current_tag is not None:
            # update position of handle
            self.current_part.update_handles_from_input(self.current_tag, pt_mouse)

            # update parameters based on handle
            self.current_part.update_params_from_handles()

            # get mask of current region (if possible)
            self.current_part.get_mask(self.image_shape)

            # update any objects that are tied to this one
            followers = self.current_part.followers
            for follower in followers:
                follower_part = getattr(self, follower)
                follower_part.update_handles_from_params()

    # update all masks
    def update_masks(self):
        for partname in self.partnames:
            getattr(self, partname).get_mask(self.image_shape)

    # draw all objects
    def draw(self, image):
        for partname in self.partnames:
            getattr(self, partname).draw(image)

    # run main window with a single image (for testing)
    def run_fake(self):
        # create an image window instance
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

        # set mouse callback
        cv2.setMouseCallback(self.name, self.on_mouse)

        # make a fake image
        image = np.zeros((640, 640, 3), dtype='uint8')
        while True:
            clone = image.copy()
            self.draw(clone)
            cv2.imshow(self.name, clone)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

    # run main window with a video capture object (cap)
    def run_video(self, cap):
        # create an image window instance
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

        # set mouse callback
        cv2.setMouseCallback(self.name, self.on_mouse)

        while True:
            # read next frame
            ret, frame = cap.read()
            if not ret:
                break

            # get shape of current image
            self.image_shape = frame.shape[:2][::-1]

            # draw lines and display image
            self.draw(frame)
            cv2.imshow(self.name, frame)

            # playback options
            k = cv2.waitKey(25) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()


# ---------------------------------------
# MAIN
# ---------------------------------------
# Run script
if __name__ == "__main__":
    # make instance of class
    fly_frame = FlyFrame()

    # # test on static image
    # main_window.run_fake()

    # test on video
    # vid_path = ('/media/sam/SamData/Mosquitoes/46_20240724/2024_07_24_0000/aedes_C001H001S0001_20240724_114750/' +
    #             'aedes_C001H001S0001_20240724_114750.avi')
    vid_path = ('/media/sam/SamData/Mosquitoes/48_20240813/other_vid/aedes_C001H001S0001_20240813_153233/' +
                'aedes_C001H001S0001_20240813_153233.avi')
    cap = cv2.VideoCapture(vid_path)
    fly_frame.run_video(cap)


