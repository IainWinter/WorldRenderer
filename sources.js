const gibsDate = (() => {
  const d = new Date(Date.now() - 36 * 3600 * 1000);
  return d.toISOString().slice(0, 10);
})();

export const IMAGERY_SOURCES = [
  {
    name: 'Esri World Imagery (sub-metre, z19)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 19,
  },
  {
    name: 'USGS National Map imagery (1 m, US only)',
    url: 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}',
    maxZoom: 16,
  },
  {
    name: 'EOX Sentinel-2 cloudless 2024 (10 m, open)',
    url: 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg',
    maxZoom: 15,
  },
  {
    name: 'EOX Sentinel-2 cloudless 2020 (10 m)',
    url: 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg',
    maxZoom: 15,
  },
  {
    name: 'NASA GIBS Blue Marble (250 m)',
    url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg',
    maxZoom: 8,
  },
  {
    name: 'NASA GIBS VIIRS true colour',
    url: `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/${gibsDate}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
    maxZoom: 9,
  },
  {
    name: 'NASA GIBS MODIS Terra',
    url: `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${gibsDate}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
    maxZoom: 9,
  },
];

export const TERRAIN_SOURCE = {
  name: 'AWS Terrain Tiles (terrarium)',
  url: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  maxZoom: 15,
};

export const PLACES = [
  { name: 'Whole earth', lon: 0, lat: 15, distance: 20000000 },
  { name: 'Mont Blanc', lon: 6.8652, lat: 45.8326, distance: 60000 },
  { name: 'Grand Canyon', lon: -112.1129, lat: 36.1069, distance: 45000 },
  { name: 'Everest', lon: 86.925, lat: 27.9881, distance: 70000 },
  { name: 'Matterhorn', lon: 7.6586, lat: 45.9763, distance: 30000 },
  { name: 'Fuji', lon: 138.7274, lat: 35.3606, distance: 50000 },
  { name: 'Andes / Aconcagua', lon: -70.0109, lat: -32.6532, distance: 70000 },
  { name: 'Iceland', lon: -19.0, lat: 64.9, distance: 400000 },
  { name: 'Antimeridian (Fiji)', lon: 179.5, lat: -17.0, distance: 300000 },
  { name: 'Himalaya wide', lon: 86.0, lat: 28.5, distance: 900000 },
];
