"""`import midas_pipeline.recon.mlem as M` must give the module (the package used to re-export a function
named `mlem`, which shadowed the submodule)."""
import inspect


def test_recon_mlem_is_the_module():
    import midas_pipeline.recon.mlem as M
    assert inspect.ismodule(M) and callable(M.mlem_recon) and M.mlem is M.mlem_recon


def test_package_level_api_unchanged():
    from midas_pipeline.recon import mlem_recon, osem_recon, osem, fbp_recon_per_grain  # noqa: F401
    import midas_pipeline.recon as R
    assert "mlem" not in R.__all__ and inspect.ismodule(R.mlem)
