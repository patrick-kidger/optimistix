from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DMN15102LS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction fitting problem (15,10) using 2-parameter Lorentzians.

    A least squares problem for fitting diffraction peaks in diamond powder data.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version of DMN15102.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-66-0
    """

    n: int = 66  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError("DMN15102LS objective function not implemented")

    def y0(self):
        raise NotImplementedError("DMN15102LS initial point not implemented")

    def args(self):
        return None

    def expected_result(self):
        raise NotImplementedError("DMN15102LS expected result not implemented")

    def expected_objective_value(self):
        raise NotImplementedError("DMN15102LS expected objective value not implemented")


# TODO: human review required
class DMN15103LS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction fitting problem (15,10) using 3-parameter Lorentzians.

    A least squares problem for fitting diffraction peaks in diamond powder data.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version of DMN15103.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-99-0
    """

    n: int = 99  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError("DMN15103LS objective function not implemented")

    def y0(self):
        raise NotImplementedError("DMN15103LS initial point not implemented")

    def args(self):
        return None

    def expected_result(self):
        raise NotImplementedError("DMN15103LS expected result not implemented")

    def expected_objective_value(self):
        raise NotImplementedError("DMN15103LS expected objective value not implemented")


# TODO: human review required
class DMN15332LS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction fitting problem (15,33) using 2-parameter Lorentzians.

    A least squares problem for fitting diffraction peaks in diamond powder data.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version of DMN15332.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-66-0
    """

    n: int = 66  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError("DMN15332LS objective function not implemented")

    def y0(self):
        raise NotImplementedError("DMN15332LS initial point not implemented")

    def args(self):
        return None

    def expected_result(self):
        raise NotImplementedError("DMN15332LS expected result not implemented")

    def expected_objective_value(self):
        raise NotImplementedError("DMN15332LS expected objective value not implemented")


# TODO: human review required
class DMN15333LS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction fitting problem (15,33) using 3-parameter Lorentzians.

    A least squares problem for fitting diffraction peaks in diamond powder data.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version of DMN15333.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-99-0
    """

    n: int = 99  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError("DMN15333LS objective function not implemented")

    def y0(self):
        raise NotImplementedError("DMN15333LS initial point not implemented")

    def args(self):
        return None

    def expected_result(self):
        raise NotImplementedError("DMN15333LS expected result not implemented")

    def expected_objective_value(self):
        raise NotImplementedError("DMN15333LS expected objective value not implemented")


# TODO: human review required
class DMN37142LS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction fitting problem (37,14) using 2-parameter Lorentzians.

    A least squares problem for fitting diffraction peaks in diamond powder data.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version of DMN37142.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-66-0
    """

    n: int = 66  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError("DMN37142LS objective function not implemented")

    def y0(self):
        raise NotImplementedError("DMN37142LS initial point not implemented")

    def args(self):
        return None

    def expected_result(self):
        raise NotImplementedError("DMN37142LS expected result not implemented")

    def expected_objective_value(self):
        raise NotImplementedError("DMN37142LS expected objective value not implemented")


# TODO: human review required
class DMN37143LS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction fitting problem (37,14) using 3-parameter Lorentzians.

    A least squares problem for fitting diffraction peaks in diamond powder data.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version of DMN37143.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-99-0
    """

    n: int = 99  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError("DMN37143LS objective function not implemented")

    def y0(self):
        raise NotImplementedError("DMN37143LS initial point not implemented")

    def args(self):
        return None

    def expected_result(self):
        raise NotImplementedError("DMN37143LS expected result not implemented")

    def expected_objective_value(self):
        raise NotImplementedError("DMN37143LS expected objective value not implemented")
