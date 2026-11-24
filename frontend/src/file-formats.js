export const SUPPORTED_EXTENSIONS = ['pdf', 'docx', 'txt', 'text', 'md', 'markdown', 'log', 'json', 'html', 'htm', 'csv', 'tsv', 'xlsx', 'xls', 'xlsm', 'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'];
export const FILE_ACCEPT = SUPPORTED_EXTENSIONS.map(extension => '.' + extension).join(',');
export function supportedFilename(name) {
  return SUPPORTED_EXTENSIONS.includes(name.split('.').pop().toLowerCase()) && name.includes('.');
}
