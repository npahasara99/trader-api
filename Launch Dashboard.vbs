Set shell = CreateObject("WScript.Shell")
repoPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run "cmd /c """ & repoPath & "\Launch Dashboard.bat""", 0, False
