class Domain:
    """
    Class to represent a domain with various parameters and boundary conditions.

    Attributes:
        number_harmonics (int): The number of harmonics in the domain.
        height (float): The height of the domain.
        radial_width (float): The radial width of the domain.
        top_BC (any): Boundary condition at the top of the domain.
        bottom_BC (any): Boundary condition at the bottom of the domain.
        category (str): The category of the domain.
    """

    def __init__(self, number_harmonics: int, height: float, radial_width: float, top_BC: any, bottom_BC: any, category: str):
        """
        Initializes the Domain class.

        Args:
            number_harmonics (int): The number of harmonics in the domain.
            height (float): The height of the domain.
            radial_width (float): The radial width of the domain.
            top_BC (any): The boundary condition at the top.
            bottom_BC (any): The boundary condition at the bottom.
            category (str): The category of the domain.
        """
        self.number_harmonics = number_harmonics
        self.height = height
        self.radial_width = radial_width
        self.top_BC = top_BC
        self.bottom_BC = bottom_BC
        self.category = category

    def radial_eigenfunctions(self, r: float):
        """
        Calculates the radial eigenfunctions at a given radial position.

        Args:
            r (float): Radial position.

        Returns:
            (float): Radial eigenfunction value.
        """
        pass

    def vertical_eigenfunctions(self, z: float):
        """
        Calculates the vertical eigenfunctions at a given vertical position.

        Args:
            z (float): Vertical position.

        Returns:
            (float): Vertical eigenfunction value.
        """
        pass

    def particular_potential(self, r: float, z: float):
        """
        Calculates the particular potential at a given (r, z) position.

        Args:
            r (float): Radial position.
            z (float): Vertical position.

        Returns:
            (float): Particular potential value.
        """
        pass
