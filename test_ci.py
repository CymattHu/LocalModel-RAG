from app import test_app

def test_run():
    result = test_app()
    assert "测试通过" in result