import pytest
import numpy as np
from pyacoustics.schema import SSPModel, SSPLayer
from pyacoustics.environment import SSPInterpolator

def test_linear_ssp():
    data = [SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1520.0)]
    model = SSPModel(type="c-linear", data=data)
    interp = SSPInterpolator(model)
    
    # Test at bounds
    c, dc, d2c = interp.evaluate(0.0)
    assert c == 1500.0
    assert dc == 0.2
    
    # Test midpoint
    c, dc, d2c = interp.evaluate(50.0)
    assert c == 1510.0
    assert dc == 0.2
    
    # Test out of bounds (currently implemented as constant clamping)
    c, dc, d2c = interp.evaluate(-10.0)
    assert c == 1500.0
    
def test_spline_ssp():
    # 3 points for a simple curve
    data = [
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=50.0, c=1490.0),
        SSPLayer(depth=100.0, c=1510.0)
    ]
    model = SSPModel(type="spline", data=data)
    interp = SSPInterpolator(model)
    
    # At exact points, it should match
    c0, _, _ = interp.evaluate(0.0)
    assert pytest.approx(c0) == 1500.0
    
    c50, _, _ = interp.evaluate(50.0)
    assert pytest.approx(c50) == 1490.0
