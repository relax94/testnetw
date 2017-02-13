import React, {Component} from 'react'
import {InputMaterial} from './InputMaterial'

export class SettingsComponent extends Component {

    render() {
        return (
            <div className="settings-panel">
                <h2>Settings Panel</h2>
                <InputMaterial name="Input Count"/>
                <InputMaterial name="Hidden Count"/>
            </div>
    )
    }

    }