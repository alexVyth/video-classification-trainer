from video_trainer.loading.fst_kinoscope import enums


class TestMapColorToCategory:
    @staticmethod
    def test_should_map_red_color_to_swimming() -> None:
        # arrange
        color = enums.Color.RED
        category_expected = enums.FstCategory.SWIMMING

        # act
        category = enums.map_color_to_category(color)

        # assert

        assert category is category_expected
