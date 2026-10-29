from eval.action4_material_cycle_probe import cycle_prefix_index

def test_explicit_prefix_and_default_end():
    assert cycle_prefix_index([4,5,6,7],6)==2
    assert cycle_prefix_index([4,5,6,7])==3
    for bad in [4,8,True]:
        try:cycle_prefix_index([4,5,6,7],bad)
        except ValueError:pass
        else:raise AssertionError("Invalid nonquery frame accepted")
