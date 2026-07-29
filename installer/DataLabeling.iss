; Inno Setup script — build after PyInstaller:
;   CPU: ISCC.exe installer\DataLabeling.iss
;   GPU: ISCC.exe /DSrcDir=..\dist_gpu\TrainingDeepAI /DEdition=GPU installer\DataLabeling.iss
#define MyAppName "TrainingDeepAI"
#define MyAppVersion "1.3.0"
#define MyAppPublisher "MVA"
#define MyAppURL "https://github.com/vietanh-ptp1411/DataLabeling"
#define MyAppExeName "TrainingDeepAI.exe"
#ifndef SrcDir
#define SrcDir "..\dist\TrainingDeepAI"
#endif
#ifndef Edition
#define Edition "CPU"
#endif
#if Edition == "GPU"
#define OutName "TrainingDeepAI-GPU-Setup-" + MyAppVersion
#else
#define OutName "TrainingDeepAI-Setup-" + MyAppVersion
#endif

[Setup]
AppId={{7E4B7A61-2C0B-4E3D-9B5A-0D8F3C1A2E77}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename={#OutName}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
