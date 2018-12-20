from behave import given, when, then
import quilt
import rasterio


@given("some view of Antarctica {bounding_box}")
def window_view_of_Antarctica(context, bounding_box):
    xmin, ymin, xmax, ymax = [float(c) for c in bounding_box.split(",")]
    context.window_bound = rasterio.coords.BoundingBox(
        left=xmin, bottom=ymin, right=xmax, top=ymax
    )


@when("we gather low and high resolution images related to that view")
def get_model_input_raster_images(context):
    # TODO refactor code below that is hardcoded for a particular test region
    if context.window_bound == rasterio.coords.BoundingBox(
        left=-1_593_714.328, bottom=-164_173.7848, right=-1_575_464.328, top=-97923.7848
    ):
        quilt.install(package="weiji14/deepbedmap/model/test", force=True)
        pkg = quilt.load(pkginfo="weiji14/deepbedmap/model/test")
        input_tiles = pkg.X_tile(), pkg.W1_tile(), pkg.W2_tile()
    else:
        input_tiles = context.deepbedmap.get_deepbedmap_model_inputs(
            window_bound=context.window_bound
        )
    context.X_tile, context.W1_tile, context.W2_tile = input_tiles


@when("pass those images into our trained neural network model")
def predict_using_trained_neural_network(context):
    model = context.deepbedmap.load_trained_model(
        model_inputs=(context.X_tile, context.W1_tile, context.W2_tile)
    )
    context.Y_hat = model.predict(
        x=[context.X_tile, context.W1_tile, context.W2_tile], verbose=0
    )


@then("a four times upsampled super resolution bed elevation map is returned")
def step_impl(context):
    # Ensure input (X_tile) and output (Y_hat) shape is like (1, height, width, 1)
    assert context.X_tile.ndim == 4
    assert context.Y_hat.ndim == 4

    # Check that High Resolution output shape (DeepBedMap) divided by
    # Low Resolution input shape (BEDMAP2) minus 2 pixel (1km) padding
    # is exactly equal to 4
    assert context.Y_hat.shape[1] / (context.X_tile.shape[1] - 2) == 4.0
    assert context.Y_hat.shape[2] / (context.X_tile.shape[2] - 2) == 4.0
