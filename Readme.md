# CapMIT1003 Database

The CapMIT1003 database contains captions and clicks collected for images from the MIT1003 database, for which reference eye scanpath are available. The database is distributed as a single SQLite3 database named `capmit1003.db`. For convenience, a lightweight Python class to access the database is provided (see "Programmatic Usage").

## Column Descriptions

| Name       | Type      | Description                                                |
|------------|-----------|------------------------------------------------------------|
| obs_uid    | String    | Unique identifier for a labeled image-caption pair.        | 
| usr_uid    | String    | Unique identifier for a single user.                       |
| start_time | Timestamp | Date and time (absolute) at which image was shown to user. |
| caption    | String    | Caption provided by the user.                              |
| img_uid    | String    | Unique identifier for a single image from MIT1003.         |
| img_path   | String    | File name of image from MIT1003.                           |
| click_id   | Integer   | Ascending identifier that may be used to order clicks.     |
| x          | Integer   | Horizontal position of click in image pixel coordinates.   |
| y          | Integer   | Vertical position of click in image pixel coordinates.     |
| click_time | Timestamp | Date and time (absolute) at which user clicked on image.   |

## Programmatic Usage

The file `capmit1003.py` provides a `CapMIT1003` dataset class to query the SQLite3 database. Its only dependency is **pandas**, a popular library for handling tabular datasets. The following snippet demonstrates how to iterate over all image-caption pairs and load the image, caption and click path. In addition, it downloads and extracts the MIT1003 stimuli images if they are not already present.

```python
from capmit1003 import CapMIT1003

CapMIT1003.download_images()
with CapMIT1003('capmit1003.db') as db:
    image_captions = db.get_captions()
    for pair in image_captions.itertuples(index=False):
        image = imread(pair.img_path)  # e.g., using Pillow, scikit-image, etc.
        caption = pair.caption
        click_path = db.get_click_path(pair.obs_uid)
        xy_coordinates = click_path[['x', 'y']].values
```

Note the usage with `with`; this ensures that the database is properly closed after all queries.

## Generate scanpaths with NevaClip

The file `main.py` provide an example to simulate scanpaths with NevaClip, using the caption encoding as guidance.

## Citation

When using the dataset, please make sure to also cite the original MIT1003 database.

```
@InProceedings{Judd_2009,
  author    = {Tilke Judd and Krista Ehinger and Fr{\'e}do Durand and Antonio Torralba},
  title     = {Learning to Predict Where Humans Look},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2009}
}
```