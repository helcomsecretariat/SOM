const { FusesPlugin } = require('@electron-forge/plugin-fuses');
const { FuseV1Options, FuseVersion } = require('@electron/fuses');
const path = require('path');
const fse  = require('fs-extra');

async function updateStartSOM() {
  const src  = path.resolve(__dirname, '../src');
  const dest = path.resolve(__dirname, 'src');
  await fse.copy(src, dest, { overwrite: true });
}

async function updateMakeSOM() {
  // copy python src directory into dist
  await fse.copy(
    path.resolve(__dirname, '../src'), 
    path.resolve(__dirname, 'out/application-win32-x64/src'), 
    { overwrite: true }
  );
  // copy pyproject.toml into dist
  await fse.copy(
    path.resolve(__dirname, '../pyproject.toml'), 
    path.resolve(__dirname, 'out/application-win32-x64/pyproject.toml'), 
    { overwrite: true }
  );
}

module.exports = {
  packagerConfig: {
    asar: true,
  },
  rebuildConfig: {},
  makers: [
    {
      name: '@electron-forge/maker-squirrel',
      config: {},
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin'],
    },
    {
      name: '@electron-forge/maker-deb',
      config: {},
    },
    {
      name: '@electron-forge/maker-rpm',
      config: {},
    },
  ],
  plugins: [
    {
      name: '@electron-forge/plugin-auto-unpack-natives',
      config: {},
    },
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: true,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
      [FuseV1Options.OnlyLoadAppFromAsar]: true,
    }),
  ],
  hooks: {
    postStart: async (forgeConfig) => {
      await updateStartSOM();
    }, 
    postMake: async (forgeConfig, makeResults) => {
      await updateMakeSOM();
    }
  }
};
