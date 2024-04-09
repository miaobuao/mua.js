export default {
  '*.{js,jsx,ts,tsx,vue,sass}': [ 'eslint --fix' ],
  '*.{html,vue,vss,less,md}': [ 'prettier --write' ],
  'package.json': [ 'prettier --write' ],
}
 