### There are 3 sub folders each containing different shapefile examples

1. **5poly** : A very simple file containing only 5 polygons with different number of sides
2. **cb_2018_us_national_20m**: Real word shapefile containing the 2018 Census Bureau US map at 20m resolution
3. **multipoly**: Test file containing multipolygons.

### Files Included in the Shapefile Dataset

1. **.shp** - **Shape Format**: Contains the geometric data (polygons) for the features in the dataset. The first 100 bytes contain metadata about the file.

2. **.shx** - **Shape Index Format**: Contains the index data for the .shp file, allowing for quick access to the geometric data.

3. **.dbf** - **Attribute Format**: Contains attribute data for the features in the .shp file. This is a dBase file that stores information in a tabular format, where each row corresponds to a feature in the .shp file.

4. **.cpg** - **Code Page Format**: Contains character encoding information for the .dbf file, ensuring that text data is correctly interpreted. Eg. ISO-8859-1
